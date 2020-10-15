import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_net import BaseNet
from ..misc import get_config_container


class DeepLIOFusionCat():
    def __init__(self, input_shapes, cfg):
        """
        :param input_shapes: the inputshape of the layers of dim = [[BxN_0],...,[BxN_i]
        :param cfg:
        """
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations
        self.type = cfg.get('type', 'cat').lower()
        self.input_shapes = input_shapes

        sum_in_channels = sum([in_shape[-1] for in_shape in self.input_shapes])
        self.output_shape = [1, self.seq_size, sum_in_channels]

    def forward(self, x):
        lidar_feat = x[0]
        imu_feat = x[1]
        if self.type == 'cat':
            out = torch.cat((lidar_feat, imu_feat), dim=2)
        else:
            raise NotImplementedError()
        return out

    def get_output_shape(self):
        return self.output_shape

    def __call__(self, x):
        return self.forward(x)


class DeepLIOFusionSoft(BaseNet):
    def __init__(self, input_shapes, cfg):
        """
        :param input_shapes: the inputshape of the layers of dim = [[BxN_0],...,[BxN_i]
        :param cfg:
        """
        super(DeepLIOFusionSoft, self).__init__()
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations
        self.input_shapes = input_shapes
        self.s1_feat = None
        self.s2_feat = None

        sum_in_channels = sum([in_shape[-1] for in_shape in self.input_shapes])

        layers = []
        for in_shape in self.input_shapes:
            b, s, n = in_shape
            layers.append(nn.Linear(sum_in_channels, n))
        self.layers = nn.ModuleList(layers)

        self.output_shape = [1, self.seq_size, sum_in_channels]

    def forward(self, x):
        lidar_feat = x[0]
        imu_feat = x[1]

        cat_feat = torch.cat((lidar_feat, imu_feat), dim=2)
        self.s1_feat = torch.sigmoid(self.layers[0](cat_feat))
        self.s2_feat = torch.sigmoid(self.layers[1](cat_feat))

        lidar_feat *= self.s1_feat
        imu_feat *= self.s2_feat
        out = torch.cat((lidar_feat, imu_feat), dim=2)
        return out

    def get_output_shape(self):
        return self.output_shape
