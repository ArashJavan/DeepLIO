import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoConstLoss(nn.Module):
    def __init__(self):
        super(GeoConstLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='sum')

    def forward(self, pred, gt):
        x = pred
        x_gt = gt[0]

        loss = self.loss_fn(x, x_gt)
        return loss

