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
        self.conv1a = nn.Sequential(nn.Conv2d(c, 64, kernel_size=3, stride=(1, 2), padding=1),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        self.conv1b = nn.Sequential(nn.Conv2d(c, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        # First block
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire2 = Fire(64, 16, 64, 64, bn=True, bn_d=bn_d)
        self.fire3 = Fire(128, 16, 64, 64, bn=True, bn_d=bn_d)
        self.se1 = SELayer(128, reduction=2)

        # second block
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire4 = Fire(128, 32, 128, 128, bn=True, bn_d=bn_d)
        self.fire5 = Fire(256, 32, 128, 128, bn=True, bn_d=bn_d)
        self.se2 = SELayer(256, reduction=2)

        # third block
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)
        self.fire6 = Fire(256, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire7 = Fire(384, 48, 192, 192, bn=True, bn_d=bn_d)
        self.fire8 = Fire(384, 64, 256, 256, bn=True, bn_d=bn_d)
        self.fire9 = Fire(512, 64, 256, 256, bn=True, bn_d=bn_d)
        self.se3 = SELayer(512, reduction=2)

        self.aspp = ASPP(512, [6, 9, 12])

        self.output_shapes = self.calc_output_shape()

    def forward(self, x):
        x_1a = self.conv1a(x)  # (H, W/2)
        x_1b = self.conv1b(x)

        ### Encoder forward
        # first fire block
        x_p1 = self.pool1(x_1a)
        x_f2 = self.fire2(x_p1)
        x_f3 = self.fire3(x_f2)
        x_se1 = self.se1(x_f3)
        if self.bypass:
            x_se1 += x_f2

        # second fire block
        x_p2 = self.pool2(x_se1)
        x_f4 = self.fire4(x_p2)
        x_f5 = self.fire5(x_f4)
        x_se2 = self.se2(x_f5)
        if self.bypass:
            x_se2 += x_f4

        # third fire block
        x_p3 = self.pool3(x_se2)
        x_f6 = self.fire6(x_p3)
        x_f7 = self.fire7(x_f6)
        if self.bypass:
            x_f7 += x_f6
        x_f8 = self.fire8(x_f7)
        x_f9 = self.fire9(x_f8)
        x_se3 = self.se3(x_f9)
        if self.bypass:
            x_se3 += x_f8

        # EL forward
        x_el = self.aspp(x_se3)
        return x_1a, x_1b, x_se1, x_se2, x_se3, x_el

    def calc_output_shape(self):
        c, h, w = self.input_shape
        input = torch.rand((1, c, h, w))
        self.eval()
        with torch.no_grad():
            x_1a, x_1b, x_se1, x_se2, x_se3, x_el = self.forward(input)
        return x_1a.shape, x_1b.shape, x_se1.shape, x_se2.shape, x_se3.shape, x_el.shape

    def get_output_shape(self):
        return self.output_shapes


class PSEncoder2(BaseNet):
    def __init__(self, input_shape, cfg, bn_d = 0.1):
        super(PSEncoder2, self).__init__()
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
        self.fire2 = Fire(64, 16, 64, 64, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire3 = Fire(128, 16, 64, 64, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire4 = Fire(128, 32, 128, 128, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.se1 = SELayer(256, reduction=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1)  # 1/8

        # second block
        self.fire5 = Fire(256, 32, 128, 128, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire6 = Fire(256, 48, 192, 192, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire7 = Fire(384, 48, 192, 192, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire8 = Fire(384, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.se2 = SELayer(512, reduction=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)  # 1/16

        # third block
        self.fire9 = Fire(512, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire10 = Fire(512, 64, 256, 256, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.se3 = SELayer(512, reduction=2)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1) # 1/32

        # third block
        self.fire11 = Fire(512, 64, 384, 384, bn=True, bn_d=bn_d, bypass=self.bypass)
        self.fire12 = Fire(768, 96, 384, 384, bn=True, bn_d=bn_d, bypass=self.bypass)
        #self.se4 = SELayer(768, reduction=2)
        #self.pool5 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1) # 1/32

        self.output_shapes = self.calc_output_shape()

    def forward(self, x):
        x_1a = self.conv1a(x)  # (H, W/2)
        x_p1 = self.pool1(x_1a)

        ### Encoder forward
        # first fire block
        x_f2 = self.fire2(x_p1)
        x_f3 = self.fire3(x_f2)
        x_f4 = self.fire4(x_f3)
        x_se1 = self.se1(x_f4)
        x_p2 = self.pool2(x_se1)

        x_f5 = self.fire5(x_p2)
        x_f6 = self.fire6(x_f5)
        x_f7 = self.fire7(x_f6)
        x_f8 = self.fire8(x_f7)
        x_se2 = self.se2(x_f8)
        x_p3 = self.pool3(x_se2)

        # second fire block
        x_f9 = self.fire9(x_p3)
        x_f10 = self.fire10(x_f9)
        x_se3 = self.se3(x_f10)
        x_p4 = self.pool4(x_se3)

        # third fire block
        x_f11 = self.fire11(x_p4)
        x_f12 = self.fire12(x_f11)
        #x_se4 = self.se4(x_f12)
        #x_p5 = self.pool5(x_se4)

        out = x_f12
        return out

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



