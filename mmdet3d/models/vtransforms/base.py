from asyncio import FastChildWatcher
from distutils.archive_util import make_archive
from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn
import numpy as np

from mmdet3d.ops import bev_pool
from mmdet3d.models.fusion_layers import apply_3d_transformation

__all__ = ["BaseTransform", "BaseDepthTransform", "BaseDepthTransfromv2"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        mask_prob: float=0.0,
        select_prob: float=1.0
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

        self.mask_prob = mask_prob
        self.select_prob = select_prob

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
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
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        img_metas,
        **kwargs,
    ):
        B = len(img_metas)
        geoms = []
        for i in range(B):
            # B x N x D x H x W x 3
            img_meta = img_metas[i]
            img_trans = img_meta['img_translation'].type_as(self.frustum)
            img_scale = img_meta['img_scale'].type_as(self.frustum)
            N = img_trans.size(0)
            D, H, W, _ = self.frustum.size()
            points = self.frustum + img_trans.view(N, 1, 1, 1, 3)
            points[..., :2] = points[..., :2] / img_scale.view(N, 1, 1, 1, 1)
            points = torch.cat([points[..., :2] * points[..., 2:], points[..., 2:], points.new_ones(N, D, H, W, 1)], dim=-1)
            img2lidar = torch.inverse(img_meta['lidar2img']).type_as(points)
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
            geoms.append(points)

        return torch.stack(geoms, dim=0)

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool.bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        img_metas,
        **kwargs
    ):

        geom = self.get_geometry(img_metas)

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        img_metas,
        **kwargs,
    ):

        batch_size = len(img_metas)
        depth = torch.zeros(batch_size, 6, 1, *self.image_size).to(img.device)
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
            #print(img_meta)
            cur_coords = cur_coords - img_trans[:, :2].view(6, 2, 1)
            cur_coords = cur_coords.transpose(1, 2)
            #dist = dist.squeeze(1)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(6):
                #print(on_img[c].sum())
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        geom = self.get_geometry(img_metas)

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x

class BaseDepthTransformv2(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        batch_size = len(points)
        depth = torch.zeros(batch_size, 6, 1, *self.image_size).to(points[0].device)
        valid = torch.zeros(batch_size, 6, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3].transpose(1, 0)
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(6):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
                N = masked_coords.shape[0]
                mask_indices = np.random.choice(N, size=int(N*self.mask_prob), replace=False)
                mask_indices = torch.from_numpy(mask_indices).long()
                coords = masked_coords[mask_indices]
                valid[b, c, 0, coords[:, 0], coords[:, 1]] = 1.0
        with torch.no_grad():
            masked_depth = (valid * depth).clone().detach()
            depth = (1 - valid) * depth


        geom = self.get_geometry(
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            extra_rots,
            extra_trans,
        )

        x, pred_depth = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x, pred_depth, valid, masked_depth