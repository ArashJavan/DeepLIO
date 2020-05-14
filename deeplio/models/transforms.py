import numbers
import torch


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        images = [torch.from_numpy(img.transpose(2, 0, 1)) for img in data[0]]
        images = torch.stack(images)
        imus = [torch.FloatTensor(d) for d in data[1]]
        gts = [torch.FloatTensor(d) for d in data[2]]
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


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data):
        images = data[0]
        _, _, image_height, image_width = images.size()
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))

        data[0] = images[:, :, crop_top:-crop_top, crop_left:-crop_left]
        return
