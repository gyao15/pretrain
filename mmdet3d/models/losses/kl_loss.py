import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class KLLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss_weight = loss_weight
    
    def forward(self, p, q, weights=None, avg_factor=None):
        loss = ((p - q) * torch.log(p / (q+1e-3) + 1e-3)).sum(dim=-2)
        if weights is not None:
            loss = loss * weights
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
        return loss