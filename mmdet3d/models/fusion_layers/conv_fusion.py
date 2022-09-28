import torch
from mmcv.cnn import xavier_init
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class ConvFusion(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of modules."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                xavier_init(m, distribution='uniform')
    
    def forward(self, pts_feats, img_feats):
        out = self.layers(torch.cat([img_feats, pts_feats], dim=1))
        return out