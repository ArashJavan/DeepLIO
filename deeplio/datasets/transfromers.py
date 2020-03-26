import torch
import torchvision.transforms.functional as F

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        imgs_0 = data['images'][0]
        imgs_1 = data['images'][1]

        data['images'] = [torch.from_numpy(imgs_0.transpose(0, 3, 1, 2)),
                          torch.from_numpy(imgs_1.transpose(0, 3, 1, 2))]
        data['imu'] = [torch.FloatTensor(data['imu'][i]) for i in range(len(data['imu']))]
        data['ground-truth'] = [torch.FloatTensor(data['ground-truth'][i]) for i in range(len(data['ground-truth']))]
        data['combinations'] = torch.FloatTensor(data['combinations'])
        return data


class Normalize:
    def __init__(self, mean, std, inplace=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data, **kwargs):
        imgs_0 = data['images'][0]
        imgs_1 = data['images'][1]

        if not self.inplace:
            imgs_0 = imgs_0.clone()
            imgs_0 = imgs_1.clone()

        dtype = imgs_0.dtype
        device = imgs_0.device

        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)

        imgs_0.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        imgs_1.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

        data['images'] = [imgs_0, imgs_1]
        return data
