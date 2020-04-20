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


class SqueezeNet(nn.Module):
    """
    SqeeuzeSeq architecture
    """
    def __init__(self, in_channels, p=0.):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride=(1, 2), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire2 = Fire(64, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 0), ceil_mode=True)

        self.p = p
        if p > 0:
            self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)

        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool3(x)

        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.fire9(x)
        x = self.pool4(x)

        if self.p > 0:
            x = self.dropout(x)
        return x







