import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=128):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.25))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class BaseBlk(nn.Module):
    def __init__(self, init='kaiming'):
        super(BaseBlk, self).__init__()
        self.init = init
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                self.init_module(module)

    def init_module(self, module):
        # Pytorch does kaiming initialiaztion at the moment, so do not need to inititliaze again.
        if 'kaiming' in self.init:
            return

        init_method = init_methods[self.init]

        if self.init == "bilinear" and isinstance(module, nn.ConvTranspose2d):
            init_method(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class Fire(BaseBlk):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, bn=True, bn_d=0.1, init='kaiming', bypass=None):
        super(Fire, self).__init__(init)
        self.inplanes = inplanes
        self.bypass = bypass
        self.bn = bn

        self.activation = nn.ReLU(inplace=False)

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        if self.bn:
            self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=bn_d)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        if self.bn:
            self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=bn_d)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        if self.bn:
            self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=bn_d)

        outplanes = expand1x1_planes + expand3x3_planes

        self.need_upsample = inplanes != outplanes
        if (self.bypass is not None) and (self.bypass == "complex") and self.need_upsample:
            self.upsample = nn.Conv2d(inplanes, outplanes, kernel_size=1)

        self.reset_parameters()

    def forward(self, x):
        identity = x
        x = self.squeeze(x)
        if self.bn:
            x = self.squeeze_bn(x)
        x = self.activation(x)

        x_1x1 = self.expand1x1(x)
        if self.bn:
            x_1x1 = self.expand1x1_bn(x_1x1)
        x_1x1 = self.activation(x_1x1)

        x_3x3 = self.expand3x3(x)
        if self.bn:
            x_3x3 = self.expand3x3_bn(x_3x3)
        x_3x3 = self.activation(x_3x3)

        out = torch.cat([x_1x1, x_3x3], 1)

        if self.bypass is not None:
            if self.bypass == "complex" and self.need_upsample:
                identity = self.upsample(identity)
                out += identity
            elif self.bypass == "simple" and not self.need_upsample:
                out += identity
            #out = F.elu(out, inplace=True)
        return out


class FireDeconv(BaseBlk):
    """Fire deconvolution layer constructor.
    Args:
      inputs: input channels
      squeeze_planes: number of 1x1 filters in squeeze layer.
      expand1x1_planes: number of 1x1 filters in expand layer.
      expand3x3_planes: number of 3x3 filters in expand layer.
      stride: spatial upsampling factors.[1,2]
    Returns:
      fire deconv layer operation.
    """
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes,
                 stride=(1, 2), padding=(0, 1), bn=True, bn_d=0.1, init='kaiming'):
        super(FireDeconv, self).__init__(init)
        self.bn = bn

        self.activation = nn.ReLU(inplace=True)

        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        if self.bn:
            self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=bn_d)

        # upsampling
        self.squeeze_deconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                                 kernel_size=(1, 4),
                                                 stride=stride, padding=padding)

        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        if self.bn:
            self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=bn_d)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        if self.bn:
            self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=bn_d)
        self.reset_parameters()

    def forward(self, x):
        x = self.squeeze(x)
        if self.bn:
            x = self.squeeze_bn(x)
        x = self.activation(x)

        x = self.activation(self.squeeze_deconv(x))

        x_1x1 = self.expand1x1(x)
        if self.bn:
            x_1x1 = self.expand1x1_bn(x_1x1)
        x_1x1 = self.activation(x_1x1)

        x_3x3 = self.expand3x3(x)
        if self.bn:
            x_3x3 = self.expand3x3_bn(x_3x3)
        x_3x3 = self.activation(x_3x3)

        out = torch.cat([x_1x1, x_3x3], 1)
        return out


class SELayer(nn.Module):
    """Squeeze and Excitation layer from SEnet
    """
    def __init__(self, in_features, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_scaled = x * y.expand_as(x)
        return x_scaled


def init_bilinear(tensor):
    """Reset the weight and bias."""
    nn.init.constant_(tensor, 0)
    in_feat, out_feat, h, w = tensor.shape

    assert h == 1, 'Now only support size_h=1'
    assert in_feat == out_feat, \
        'In bilinear interporlation mode, input channel size and output' \
        'filter size should be the same'
    factor_w = (w + 1) // 2

    if w % 2 == 1:
        center_w = factor_w - 1
    else:
        center_w = factor_w - 0.5

    og_w = torch.reshape(torch.arange(w), (h, -1))
    up_kernel = (1 - torch.abs(og_w - center_w) / factor_w)
    for c in range(in_feat):
        tensor.data[c, c, :, :] = up_kernel


init_methods = {
    'xavier-normal': nn.init.xavier_normal_,
    'xavier-uniform': nn.init.xavier_uniform_,
    'kaiming': nn.init.kaiming_normal_,
    'bilinear': init_bilinear,
}