import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplio.common.spatial import normalize_quaternion

class GeoConstLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(GeoConstLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='sum')
        self.alpha = alpha

    def forward(self, pred, gt):
        x_hat = pred[0]
        x_gt = gt[0]

        q_hat = normalize_quaternion(pred[1])
        q_gt = gt[1]

        loss = self.alpha * self.loss_fn(x_hat, x_gt) + (1 - self.alpha) * self.loss_fn(q_hat, q_gt)
        return loss

