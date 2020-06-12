import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """
    Basenet for all modules
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.pretrained = False

    def get_output_shape(self):
        raise NotImplementedError()

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


class BaseDeepLIO(BaseNet):
    """
    Base network for just main modules, e.g. deepio, deeplo and deeplios
    """
    def get_feat_networks(self):
        raise NotImplementedError()

    def initialize(self):
        raise NotImplementedError()


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
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
