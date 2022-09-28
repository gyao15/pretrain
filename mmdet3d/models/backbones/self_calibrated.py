from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import load_checkpoint
from torch import nn as nn
from torch.nn import functional as F
import torch

from mmdet.models import BACKBONES

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, pooling_r, norm_cfg):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, bias=False),
                    build_norm_layer(norm_cfg, planes)[1],
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, bias=False),
                    build_norm_layer(norm_cfg, planes)[1],
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, bias=False),
                    build_norm_layer(norm_cfg, planes)[1],
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCBottleneck(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SCBottleneck, self).__init__()
        group_width = int(planes / self.expansion)
        #self.conv1_a = build_conv_layer(conv_cfg, inplanes, group_width, 1, stride=1)
        #self.bn1_a = build_norm_layer(norm_cfg, group_width)[1]
        self.conv1_b = build_conv_layer(conv_cfg, inplanes, group_width, 1, stride=1)
        self.bn1_b = build_norm_layer(norm_cfg, group_width)[1]

        #self.k1 = nn.Sequential(
        #            build_conv_layer(conv_cfg,
        #                group_width, group_width, 3, stride=stride,
        #                padding=1),
        #            build_norm_layer(norm_cfg, group_width)[1],
        #            )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=1, pooling_r=self.pooling_r, norm_cfg=norm_cfg)

        self.conv3 = build_conv_layer(conv_cfg,
            group_width, planes, 1, stride=1)
        self.bn3 = build_norm_layer(norm_cfg, planes)[1]

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        #print(x.shape)

        #out_a= self.conv1_a(x)
        #out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        #out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        #print(out_a.shape, out_b.shape)

        #out_a = self.k1(out_a)
        out_b = self.scconv(out_b)
        #out_a = self.relu(out_a)
        out_b = self.relu(out_b)
        #print(out_a.shape, out_b.shape)

        out = self.conv3(out_b)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class SCNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=[128, 256], layer_nums=[5, 5], layer_strides=[2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False)):
        super(SCNet, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]

        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    SCBottleneck(out_channels[i], out_channels[i], stride=1, norm_cfg=norm_cfg, conv_cfg=conv_cfg))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)