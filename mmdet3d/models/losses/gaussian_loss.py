import torch
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class GaussianLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(GaussianLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_mean, preds_var, targets, weights=None, avg_factor=None):
        loss = (preds_mean - targets).pow(2) / (2 * torch.clamp(preds_var.exp().pow(2), min=1e-4)) + torch.clamp(preds_var, min=-8)
        #print(preds_var.exp())
        #loss = (loss).sum(-1)
        if weights is not None:
            loss = loss * weights
        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        else:
            loss = loss.mean()
        return loss
