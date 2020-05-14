import torch
import torch.nn as nn


class BaseNet(nn.Module):
    """
    DeepLIO with Simple Siamese SqueezeNet
    """
    def __init__(self, input_shape, cfg):
        super(BaseNet, self).__init__()
        self.p = cfg['dropout']

        self.input_shape = input_shape

        # number of channels, width and height
        self.h, self.w, self.c = input_shape

    def conv_forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


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
