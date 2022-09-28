import copy
import numpy as np
import torch
import cv2
import mmcv
import torch.nn as nn
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.functions import MSDeformAttnFunction
from mmcv.cnn import xavier_init, constant_init
from torch.nn.init import xavier_uniform_, constant_
import math

class SinePositionalEncoding(nn.Module):

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def get_reference_points(H, W, Z=8, num_points_in_pillar=4, batch_size=1):
    zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(batch_size, 1, 1, 1)
    return ref_3d

def lidar2img(points, img_meta):
    lidar2img_rt = img_meta['lidar2img'].type_as(points)
    img_scale_factor = points.new_tensor(img_meta['scale_factor'][:2] if 'scale_factor' in img_meta else [1.0, 1.0])
    img_flip = img_meta['flip'] if 'flip' in img_meta else False
    img_crop_offset = points.new_tensor(img_meta['img_crop_offset'] if 'img_crop_offset' in img_meta else 0)
    img_shape = img_meta['img_shape'][:2]
    img_pad_shape = img_meta['input_shape'][:2]
    Z, N = points.shape[:2]
    num_views = lidar2img_rt.shape[0]

    img_points = apply_3d_transformation(points.view(-1, 3), 'LIDAR', img_meta, reverse=True).detach()
    img_points = torch.cat([img_points, torch.ones_like(img_points[:, :1])], dim=-1)
    img_points = img_points.view(1, N*Z, 4).repeat(num_views, 1, 1).unsqueeze(-1)
    rt = lidar2img_rt.view(num_views, 1, 4, 4).repeat(1, N*Z, 1, 1)
    img_points = torch.matmul(rt.to(torch.float32), img_points.to(torch.float32)).squeeze(-1)

    eps = 1e-5
    bev_mask = (img_points[..., 2:3] > eps)
    img_points[..., :2] /= torch.clamp(img_points[..., 2:3], min=eps)
    img_coords = img_points[..., :2] * img_scale_factor
    img_coords -= img_crop_offset
    img_coords[..., 0] = img_coords[..., 0] / img_pad_shape[1]
    img_coords[..., 1] = img_coords[..., 1] / img_pad_shape[0]
    bev_mask = (bev_mask & (img_coords[..., :1] > 0) & (img_coords[..., :1] < 1) & (img_coords[..., 1:2] > 0) & (img_coords[..., 1:2] < 1))
    bev_mask = bev_mask.new_tensor(np.nan_to_num(bev_mask.cpu().numpy()))

    img_coords = img_coords.view(num_views, Z, N, -1).transpose(1, 2)
    bev_mask = bev_mask.view(num_views, Z, N).transpose(1, 2)
    return img_coords[..., :2], bev_mask

class Image2BEV(nn.Module):
    def __init__(self, embed_dims, dropout, num_views, num_levels, use_view_emb, bev_cfg):
        super(Image2BEV, self).__init__()
        self.embed_dims = embed_dims
        self.num_views = num_views
        self.num_levels = num_levels
        self.use_view_embed = use_view_emb
        self.bev_cfg = bev_cfg
        self.pc_range = bev_cfg['pc_range']
        self.voxel_size = bev_cfg['voxel_size']
        self.stride = bev_cfg['stride']
        self.bev_W = (self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0] / self.stride
        self.bev_H = (self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1] / self.stride

        self.bev_embedding = nn.Embedding(self.bev_W*self.bev_H, embed_dims)
        if num_levels > 1:
            self.level_embeds = nn.Parameter(torch.FloatTensor(num_levels, embed_dims))
        if use_view_emb:
            self.view_embeds = nn.Parameter(torch.FloatTensor(num_views, embed_dims))

        # self.pos_embed_layer = SinePositionalEncoding(embed_dims // 2)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def get_references_lidar_points(self, batch_size):
        Z = self.pc_range[5] - self.pc_range[2]

        ref_3d = get_reference_points(self.bev_H, self.bev_W, Z, self.bev_cfg['num_points_in_pillar'], batch_size)
        ref_3d[..., :1] = ref_3d[..., :1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., :1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        return ref_3d

    def forward(self, mlmv_feats, img_metas, ref_bev=None):
        
        feat_flatten = []
        spatial_shapes = []
        for l, feat in enumerate(mlmv_feats):
            B, _, _, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_views, bs, H*W, C
            if self.use_view_embed:
                feat = feat + self.view_embeds[:, None, None, :]
            if self.num_levels > 1:
                feat = feat + self.level_embeds[None, None, l:l+1, :]
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, dim=2)
        spatial_shapes = mlmv_feats[0].new_tensor(spatial_shapes)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.transpose(0, 1)  # bs, num_views, N, C
        if ref_bev is None:
            bev_queries = self.bev_embedding.unsqueeze(0).repeat(B, 1, 1)
        else:
            bev_queries = ref_bev
        # mask = feat_flatten.new_zeros(B, self.bev_H, self.bev_W)
        # bev_pos = self.pos_embed_layer(mask)

        ref_3d = self.get_references_lidar_points(B)
        ref_3d = ref_3d.type_as(feat_flatten)

        ref_2d = []
        bev_masks = []
        indices = []
        max_len = 0
        for i in range(B):
            img_coords, bev_mask = lidar2img(ref_3d[i], img_metas[i])
            ref_2d.append(img_coords)
            indices_per_batch = []
            bev_masks.append(bev_mask)
            for j in range(self.num_views):
                indices_per_img = bev_mask[j].sum(-1).nonzero().squeeze(-1)
                indices_per_batch.append(indices_per_img)
                if len(indices_per_batch) > max_len:
                    max_len = len(indices_per_batch)
            indices.append(indices_per_batch)
        #ref_2d = torch.stack(ref_2d, dim=1) # num_views, bs, H*W, Z, 2
        bev_masks = torch.stack(bev_masks, dim=0) # bs, num_views, H*W, Z
        
        queries_rebatch = bev_queries.new_zeros(B, self.num_views, max_len, self.embed_dims)
        ref_points = ref_2d.new_zeros(B, self.num_views, max_len, ref_2d.shape[-2], 2)

        for i in range(B):
            for j in range(self.num_views):
                index = indices[i][j]
                queries_rebatch[i, j, :len(index)] = bev_queries[i, index]
                ref_points[i, j, :len(index)] = ref_2d[i][j, index]
        
        N = feat_flatten.shape[2]
        queries_rebatch = queries_rebatch.view(B*self.num_views, max_len, self.embed_dims)
        feat_flatten = feat_flatten.view(B*self.num_views, N, self.embed_dims)
        ref_points = ref_points.view(B*self.num_views, max_len, ref_2d.shape[-2], 2)

        out = self.deformable_attention(queries_rebatch, feat_flatten, feat_flatten, ref_points, spatial_shapes, level_start_index)
        out = out.view(B, self.num_views, max_len, self.embed_dims)
        slots = torch.zeros_like(bev_queries)
        for i in range(B):
            for j in range(self.num_views):
                slots[i, indices[i][j]] += out[i, j, :len(indices[i][j])]
        count = (bev_masks.sum(-1) > 0).sum(1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return bev_queries + self.dropout(slots)

class DeformableAttention3D(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels, num_points, dropout=0.1):
        super().__init__()
        self.im2col_step = 64
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads*num_levels*num_points*2)
        self.attn_weights = nn.Linear(embed_dims, num_heads*num_levels*num_points)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        #self.out_proj = nn.Linear(embed_dims, embed_dims)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        #xavier_uniform_(self.output_proj.weight.data)
        #constant_(self.output_proj.bias.data, 0.)

    
    def forward(self, query, key, value, ref_points, spatial_shapes, level_start_idx):
        #identity = query

        B, N, _ = query.shape
        M = key.shape[1]
        v = self.v_proj(value)
        sampling_offset = self.sampling_offsets(query).view(B, N, self.num_heads, self.num_levels, self.num_points, 2)
        attn_weights = self.attn_weights(query).view(B, N, self.num_heads, self.num_levels*self.num_points)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(B, N, self.num_heads, self.num_levels, self.num_points)

        sampling_offset[..., 0] = sampling_offset[..., 0] / spatial_shapes[None, None, None, :, None, 1]
        sampling_offset[..., 1] = sampling_offset[..., 1] / spatial_shapes[None, None, None, :, None, 0]
        sampling_offset = sampling_offset.view(B, N, self.num_heads, self.num_levels, -1, ref_points.shape[-2], 2)
        sampling_locs = ref_points[:, :, None, None, None, :, :] + sampling_offset
        sampling_locs = sampling_locs.view(B, N, self.num_heads, self.num_levels, self.num_points, 2)

        output = MSDeformAttnFunction.apply(v, spatial_shapes, level_start_idx, sampling_locs, attn_weights, self.im2col_step)
        #output = self.out_proj(output)
        return output



        








        


        