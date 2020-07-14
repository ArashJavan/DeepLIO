import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_net import BaseNet
from .pointseg_modules import Fire, FireDeconv, SELayer, ASPP


class PSEncoder(BaseNet):
    def __init__(self, input_shape, cfg, bn_d = 0.1):
        super(PSEncoder, self).__init__()
        bn_d = bn_d
        self.bypass = cfg['bypass']
        self.input_shape = input_shape
        c, h, w = self.input_shape

        ### Ecnoder part
        self.conv1a = nn.Sequential(nn.Conv2d(c, 64, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2)),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)   # 1/4

        # First block
        self.fire_blk1 = nn.Sequential(
            Fire(64, 16, 64, 64, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(128, 16, 64, 64, bn=True, bn_d=bn_d, bypass=self.bypass),
            SELayer(128, reduction=2),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1))  # 1/8

        # second block
        self.fire_blk2 =nn.Sequential(
            Fire(128, 32, 128, 128, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(256, 32, 128, 128, bn=True, bn_d=bn_d, bypass=self.bypass),
            SELayer(256, reduction=2),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1))  # 1/16

        self.fire_blk3 =nn.Sequential(
            Fire(256, 48, 192, 192, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(384, 48, 192, 192, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(384, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(512, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass),
            SELayer(512, reduction=2),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1))  # 1/16

        # third block
        self.fire_blk4 = nn.Sequential(
            Fire(512, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(512, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass),
            SELayer(512, reduction=2),
            nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)) # 1/32

        # third block
        self.fire_blk5 = nn.Sequential(
            Fire(512, 80, 384, 384, bn=True, bn_d=bn_d, bypass=self.bypass),
            Fire(768, 80, 384, 384, bn=True, bn_d=bn_d, bypass=False))
            #SELayer(768, reduction=2),
            #nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)) # 1/32

        self.output_shapes = self.calc_output_shape()

    def forward(self, x):
        x_1a = self.conv1a(x)  # (H, W/2)
        x_p1 = self.pool1(x_1a)

        ### Encoder forward
        x = self.fire_blk1(x_p1)
        x = self.fire_blk2(x)
        x = self.fire_blk3(x)
        x = self.fire_blk4(x)
        x = self.fire_blk5(x)
        return x

    def calc_output_shape(self):
        c, h, w = self.input_shape
        input = torch.rand((1, c, h, w))
        self.eval()
        with torch.no_grad():
           x_se3 = self.forward(input)
        return x_se3.shape

    def get_output_shape(self):
        return self.output_shapes



class PSDecoder(BaseNet):
    def __init__(self, input_shape, cfg):
        super(PSDecoder, self).__init__()
        bn_d = 0.1
        num_classes = len(cfg['classes'])
        self.input_shapes = input_shape
        self.p = cfg['dropout']

        self.fdeconv_el = FireDeconv(128, 32, 128, 128, bn=True, bn_d=bn_d)

        self.fdeconv_1 = FireDeconv(512, 64, 128, 128, bn=True, bn_d=bn_d)
        self.fdeconv_2 = FireDeconv(512, 64, 64, 64, bn=True, bn_d=bn_d)
        self.fdeconv_3 = FireDeconv(128, 16, 32, 32, bn=True, bn_d=bn_d)
        self.fdeconv_4 = FireDeconv(64, 16, 32, 32, bn=True, bn_d=bn_d)

        self.drop = nn.Dropout2d(p=self.p)
        self.conv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        x_1a, x_1b, x_se1, x_se2, x_se3, x_el = x
        x_el = self.fdeconv_el(x_el)

        ### Decoder forward
        x_fd1 = self.fdeconv_1(x_se3)  # (H, W/8)
        x_fd1_fused = torch.add(x_fd1, x_se2)
        x_fd1_fused = torch.cat((x_fd1_fused, x_el), dim=1)

        x_fd2 = self.fdeconv_2(x_fd1_fused)  # (H, W/4)
        x_fd2_fused = torch.add(x_fd2, x_se1)

        x_fd3 = self.fdeconv_3(x_fd2_fused)  # (H, W/2)
        x_fd3_fused = torch.add(x_fd3, x_1a)

        x_fd4 = self.fdeconv_4(x_fd3_fused)  # (H, W/2)
        x_fd4_fused = torch.add(x_fd4, x_1b)

        x_d = self.drop(x_fd4_fused)
        x = self.conv2(x_d)
        return x

    def calc_output_shape(self):
        input = [torch.rand(in_shape) for in_shape in self.input_shapes]
        self.eval()
        with torch.no_grad():
            out = self.forward(input)
        return out.shape

    def get_output_shape(self):
        return self.output_shape



