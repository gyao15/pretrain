import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_upsample_layer
import mmcv

from mmdet3d.models.registry import VTRANSFORMS
from mmdet3d.models.fusion_layers import apply_3d_transformation

@VTRANSFORMS.register_module()
class StatisticalLayer(nn.Module):
    def __init__(self, channels, class_id, dist, iou, update_interval, momentum, use_statistical, preload_path=None):
        super(StatisticalLayer, self).__init__()
        self.channels = channels
        self.class_id = class_id
        self.dist = dist
        self.iou = iou
        

        
        
        if self.use_statistical:
