import torch
import torchvision.transforms.functional as F

import time

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        images = torch.stack(data[0]).permute(0, 3, 1, 2)
        imus = torch.FloatTensor(data[1])
        gts = torch.FloatTensor(data[2])
        return images, imus, gts


class Normalize:
    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        images = data[0]

        if not self.inplace:
            images = images.clone()

        dtype = images.dtype
        device = images.device

        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)

        images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

        return images, data[1], data[2]
