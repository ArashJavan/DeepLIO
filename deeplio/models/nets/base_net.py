import torch
import torch.nn as nn

from ..misc import get_config_container

class BaseNet(nn.Module):
    """
    Basenet for all modules
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.pretrained = False
        self.output_shape = None

    def get_output_shape(self):
        return self.output_shape

    @property
    def name(self):
        return self.__class__.__name__.lower()

    @property
    def device(self):
        devices = ({param.device for param in self.parameters()} |
                   {buf.device for buf in self.buffers()})
        if len(devices) != 1:
            raise RuntimeError('Cannot determine device: {} different devices found'
                               .format(len(devices)))
        return next(iter(devices))

    def get_modules(self):
        return [self]


def num_flat_features(x, dim=1):
    size = x.size()[dim:]  # all dimensions except the dim (e.g. dim=1 batch, dim=2 seq. )
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def eval_output_size_detection(model, input_shape):
    # in-feature size autodetection
    model.eval()
    c, h, w = input_shape
    with torch.no_grad():
        x = torch.randn((1, c, h, w))
        x = model(x)
        _, c, h, w = x.shape
    return c, h, w



def conv( batch_norm, in_planes, out_planes, kernel_size=(3, 3), stride=1):
    padding_h = (kernel_size[0] - 1) // 2
    padding_w = (kernel_size[1] - 1) // 2

    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(padding_h, padding_w),
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(padding_h, padding_w),
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )