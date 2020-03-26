""" Squeeze Net Model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze(x)
        x = self.squeeze_activation(x)

        x_1x1 = self.expand1x1(x)
        x_1x1 = self.expand1x1_activation(x_1x1)

        x_3x3 = self.expand3x3(x)
        x_3x3 = self.expand3x3_activation(x_3x3)

        out = torch.cat([x_1x1, x_3x3], 1)
        return out







