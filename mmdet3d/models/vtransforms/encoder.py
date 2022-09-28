from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_upsample_layer

from mmdet3d.models.registry import VTRANSFORMS
from mmdet3d.models.fusion_layers import apply_3d_transformation
from torchvision.models.resnet import Bottleneck
from typing import List

def generate_grid(h, w):
    x = torch.linspace(0, 1, int(w))
    y = torch.linspace(0, 1, int(h))
    indices = torch.stack(torch.meshgrid(x, y), dim=0)
    indices = F.pad(indices, (0,0,0,0,0,1), value=1)
    indices = indices.unsqueeze(0)
    return indices

def get_view_matrix(h=180, w=180, h_meters=108.0, w_meters=108.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return [[0., -sw, w/2.], [-sh, 0., h*offset+h/2.], [0, 0., 1.]]

class BEVEmbedding(nn.Module):
    def __init__(self,  bev_h, bev_w, h_meters, w_meters, offset, downsamples):
        super().__init__()

        h = int(bev_h // downsamples)
        w = int(bev_w // downsamples)

        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_w * grid[0]
        grid[1] = bev_h * grid[1]

        V = get_view_matrix(bev_h, bev_w, h_meters, w_meters, offset)
        V_inv = torch.FloatTensor(V).inverse()
        grid = V_inv @ grid.view(3, -1)
        grid = grid.view(3, h, w)

        self.register_buffer('grid', grid, persistent=False)
        #self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w), requires_grad=True)

    #def get_prior(self):
    #    return self.learned_features

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, dim_head * heads, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, dim_head * heads, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, dim_head * heads, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
    
    def forward(self, q, k, v, skip=None):
        B, N, D, bev_H, bev_W = q.shape
        img_H, img_W = k.shape[-2:]

        q = q.view(B, N, D, -1).permute(0, 1, 3, 2)
        k = k.view(B, N, D, -1).permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, D)

        q = self.to_q(q)
        q = q.view(B, N, -1, self.heads, self.dim_head)
        k = self.to_k(k)
        k = k.view(B, N, -1, self.heads, self.dim_head)
        v = self.to_v(v)
        v = v.view(B, -1, self.heads, self.dim_head)

        prob = self.scale * torch.einsum('bnqmd,bnkmd->bnqkm', q, k)
        prob = prob.transpose(1, 2).contiguous().view(B, bev_H*bev_W, -1, self.heads)
        att = prob.softmax(dim=-2)

        a = torch.einsum('bqkm,bkmd->bqmd', att, v)
        a = a.reshape(B, -1, self.heads*self.dim_head)
        z = self.proj(a)
        if skip is not None:
            z = z + skip.view(B, D, -1).transpose(2, 1)
        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = z.view(B, bev_H, bev_W, -1).permute(0, 3, 1, 2).contiguous()
        return z

class CrossViewAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, feature_size, img_size, dbound, qkv_bias, heads=4, dim_head=32, skip=True):
        super().__init__()

        self.img_size = img_size
        self.feature_size = feature_size
        self.dbound = dbound

        frustum = self.create_frustum()
        self.register_buffer('frustum', frustum, persistent=False)
        self.linear = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU())
        self.proj = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU())

        self.bev_embed = nn.Conv2d(2, hidden_channels, 1)
        self.frustum_embed = nn.Conv2d(3, hidden_channels, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, hidden_channels, 1, bias=False)

        self.cross_attn = CrossAttention(hidden_channels, heads, dim_head, qkv_bias)
        self.skip = skip
    
    def create_frustum(self):
        iH, iW = self.img_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def forward(self, bev_features, img_features, depth, bev_grid, img_metas):
        B, N, C, H, W = img_features.shape
        D = self.frustum.shape[0]
        _, _, bH, bW = bev_features.shape
        cam_trans = []
        frustums = []
        for i in range(B):
            img_meta = img_metas[i]
            img_trans = img_meta['img_translation'].type_as(self.frustum)
            img_scale = img_meta['img_scale'].type_as(self.frustum)
            points = self.frustum + img_trans.view(N, 1, 1, 1, 3)
            points[..., :2] = points[..., :2] / img_scale.view(N, 1, 1, 1, 1)
            points = torch.cat([points[..., :2] * points[..., 2:], points[..., 2:], points.new_ones(N, D, H, W, 1)], dim=-1)
            img2lidar = torch.inverse(img_meta['lidar2img']).type_as(points)
            cam_trans.append(img2lidar[:, :, -1:])
            points = torch.einsum('nij,n...j->n...i', [img2lidar, points])
            points = points[..., :3]
            #print(points.view(N, -1, 3).max(1), points.view(N, -1, 3).min(1))

            if self.training:
                pcd_rot = img_meta['pcd_rotation'].type_as(points)
                points = torch.matmul(points.unsqueeze(-2), pcd_rot).squeeze(-2)
                pcd_scale = img_meta['pcd_scale_factor']
                points = points * pcd_scale
                pcd_trans = img_meta['pcd_trans'].type_as(points)
                points = points + pcd_trans
                if img_meta['pcd_horizontal_flip']:
                    points[..., 1] = - points[..., 1]
                if img_meta['pcd_vertical_flip']:
                    points[..., 0] = - points[..., 0]
            
            frustums.append(points)
        cam_trans = torch.cat(cam_trans, dim=0)
        frustums = torch.stack(frustums, dim=0)
        frustums = (depth.unsqueeze(-1) * frustums).sum(dim=2)

        cam_embed = self.cam_embed(cam_trans.unsqueeze(-1))
        frustums_embed = self.frustum_embed(frustums.view(B*N, H, W, 3).permute(0, 3, 1, 2))
        img_embed = frustums_embed - cam_embed
        img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)

        bev_embed = self.bev_embed(bev_grid.unsqueeze(0))
        bev_embed = bev_embed - cam_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        bev_embed = bev_embed.view(B, N, -1, bH, bW)

        img_feats = self.proj(img_features.view(B*N, C, H, W))
        k = (img_feats + img_embed).view(B, N, -1, H, W)
        v = self.linear(img_features.view(B*N, C, H, W)).view(B, N, -1, H, W)

        q = bev_features.unsqueeze(1) + bev_embed
        return self.cross_attn(q, k, v, skip=bev_features if self.skip else None)

@VTRANSFORMS.register_module()
class Encoder(nn.Module):
    def __init__(
        self,
        img_channels,
        bev_channels,
        feature_size,
        img_size,
        xbound,
        ybound,
        dbound,
        downsamples,
        qkv_bias,
        heads=4,
        dim_head=32,
        skip=True,
        use_semantic=False,
    ):
        super().__init__()
        self.img_size = img_size
        bev_h = (xbound[1] - xbound[0]) // xbound[2]
        bev_w = (ybound[1] - ybound[0]) // ybound[2]
        h = int(bev_h // downsamples)
        w = int(bev_w // downsamples)

        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_w * grid[0]
        grid[1] = bev_h * grid[1]

        self.D = int((dbound[1] - dbound[0]) // dbound[2])
        V = get_view_matrix(bev_h, bev_w, xbound[1] - xbound[0], ybound[1] - ybound[0], offset=0)
        V_inv = torch.FloatTensor(V).inverse()
        grid = V_inv @ grid.view(3, -1)
        grid = grid.view(3, h, w)

        self.register_buffer('grid', grid, persistent=False)

        self.attn_layer = CrossViewAttention(img_channels, bev_channels, feature_size, img_size, dbound, qkv_bias, heads, dim_head, skip)

        self.forward_layer = nn.Sequential(*[Bottleneck(bev_channels, bev_channels//4) for _ in range(2)])

        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        extra_channels = 64
        self.use_semantic = use_semantic
        if self.use_semantic:
            self.stransfrom = nn.Sequential(
                nn.Conv2d(10, 16, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 5, stride=4, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
            )
            extra_channels += 64
        self.depthnet = nn.Sequential(
            nn.Conv2d(img_channels + extra_channels, extra_channels, 3, padding=1),
            nn.BatchNorm2d(extra_channels),
            nn.ReLU(True),
            nn.Conv2d(extra_channels, extra_channels, 3, padding=1),
            nn.BatchNorm2d(extra_channels),
            nn.ReLU(True),
            nn.Conv2d(extra_channels, self.D, 1),
        )
        if downsamples == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(bev_channels, bev_channels, 2*downsamples - 1, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    bev_channels,
                    bev_channels,
                    3,
                    stride=downsamples,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
            )
            upsample_cfg = dict(type='deconv', bias=False)
            upsample = build_upsample_layer(
                upsample_cfg,
                in_channels=bev_channels,
                out_channels=bev_channels,
                stride=downsamples,
                kernel_size=downsamples,
            )
            self.upsample = nn.Sequential(upsample, nn.BatchNorm2d(bev_channels, eps=0.001, momentum=0.01), nn.ReLU())
        elif downsamples == 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    bev_channels,
                    bev_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    bev_channels,
                    bev_channels,
                    3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
                nn.Conv2d(bev_channels, bev_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(True),
            )
            upsample_cfg = dict(type='deconv', bias=False)
            upsample = build_upsample_layer(
                upsample_cfg,
                in_channels=bev_channels,
                out_channels=bev_channels,
                stride=downsamples,
                kernel_size=downsamples,
            )
            self.upsample = nn.Sequential(upsample, nn.BatchNorm2d(bev_channels, eps=0.001, momentum=0.01), nn.ReLU())
        else:
            self.downsample = nn.Identity()
            self.upsample = nn.Identity()

    def get_depth(self, x, d, s=None):
        B, N, C, H, W = x.shape
        d = d.view(B*N, *d.shape[2:])
        x = x.view(B*N, C, H, W)
        d = self.dtransform(d)
        x = torch.cat([x, d], dim=1)
        if self.use_semantic and s is not None:
            s = s.view(B*N, *s.shape[2:])
            s = self.stransfrom(s)
            x = torch.cat([x, s], dim=1)
        x = self.depthnet(x)
        depth = x.softmax(dim=1)
        depth = depth.view(B, N, self.D, H, W)
        return depth

    def forward(self, img, bev, img_metas, **kwargs):
        batch_size = len(img_metas)
        depth = torch.zeros(batch_size, 6, 1, *self.img_size).to(img.device)
        if kwargs.get('points', None) is not None:
            points = kwargs['points']
        else:
            points = kwargs['raw_points']
        
        for b in range(batch_size):
            img_meta = img_metas[b]
            p = points[b][:, :3]
            if self.training and (kwargs.get('points', None) is not None):
                p = apply_3d_transformation(p, 'LIDAR', img_meta, reverse=True).detach()
            
            lidar2img = img_meta['lidar2img'].type_as(p)
            p = torch.cat([p, p.new_ones(p.size(0), 1)], dim=1)
            cur_coords = torch.matmul(lidar2img, p.transpose(1, 0))[:, :3, :]
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], min=1e-5)
            cur_coords = cur_coords[:, :2, :] / cur_coords[:, 2:, :]

            # imgaug
            img_scale = img_meta['img_scale'].type_as(p)
            cur_coords = cur_coords * img_scale.view(6, 1, 1)
            img_trans = img_meta['img_translation'].type_as(p)
            #print(img_scale, img_trans)
            cur_coords = cur_coords - img_trans[:, :2].view(6, 2, 1)
            cur_coords = cur_coords.transpose(1, 2)
            #dist = dist.squeeze(1)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.img_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.img_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(6):
                #print(on_img[c].sum())
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        depth_preds = self.get_depth(img, depth, kwargs.get('semantic', None))

        if isinstance(bev, list):
            bev = self.downsample(bev[0])
            fusion_feats = self.attn_layer(bev, img, depth_preds, self.grid[:2], img_metas)
            fusion_feats = self.forward_layer(fusion_feats)
            fusion_feats = self.upsample(fusion_feats)
            return [fusion_feats]
        else:
            bev = self.downsample(bev)
            fusion_feats = self.attn_layer(bev, img, depth_preds, self.grid[:2], img_metas)
            fusion_feats = self.forward_layer(fusion_feats)
            fusion_feats = self.upsample(fusion_feats)
            return fusion_feats


