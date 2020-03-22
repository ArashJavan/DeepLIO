import torch


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        imgs_0 = data['images'][0]
        imgs_1 = data['images'][1]

        data['images'] = [torch.from_numpy(imgs_0.transpose(0, 3, 1, 2)),
                          torch.from_numpy(imgs_1.transpose(0, 3, 1, 2))]
        data['imu'] = [torch.FloatTensor(data['imu'][i]) for i in range(len(data['imu']))]
        data['ground-truth'] = [torch.FloatTensor(data['ground-truth'][i]) for i in range(len(data['ground-truth']))]
        return data
