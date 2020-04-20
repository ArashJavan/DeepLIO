import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

from deeplio.common.laserscan import LaserScan
from deeplio.datasets.kitti import KittiRawData


class NormalNetPCA(nn.Module):
    def __init__(self, radius=0.4, max_nn=50, config=None):
        super(NormalNetPCA, self).__init__()
        self.radius = radius
        self.max_nn = max_nn

        if config is None:
            print("Error: For normal estimation using PCA we need some configurations!")
            return
        self.cfg = config
        self.image_width = config['image-width']
        self.image_height = config['image-height']
        self.fov_up = config['fov-up']
        self.fov_down = config['fov-down']

    def forward(self, x):
        points = x[:, :, 0:3].reshape(-1, 3) * KittiRawData.MAX_DIST_HDL64
        indices = np.where(np.all(points == [0., 0., 0.], axis=1))[0]
        points = np.delete(points, indices, axis=0)

        ls = LaserScan(H=self.image_height, W=self.image_width, fov_up=self.fov_up, fov_down=self.fov_down)
        ls.set_points(points)
        ls.do_range_projection()
        ls.do_normal_projection()

        normals = ls.proj_normals
        return normals