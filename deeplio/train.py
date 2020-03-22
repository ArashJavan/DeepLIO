import yaml

import torch
from torchvision import transforms

from deeplio.datasets import kitti
from deeplio.models import deeplio_nets
from deeplio.datasets import transfromers
from deeplio.losses.losses import *
from deeplio.common.spatial import *
from deeplio.visualization.utilities import *


def train(model, train_loader, epoch):
    model.train()

    for batch_index, data in enumerate(train_loader):

        # skip unvalid data without ground-truth
        if len(data['ground-truth']) == 1 and data['ground-truth'][0] == 0:
            continue

        images = [data['images'][0][0], data['images'][1][0]]
        imu = data['imu']
        gt = [data['ground-truth'][i][0][-1] for i in range(len(data['ground-truth']))]
        gt_pos = [T[:3, 3] for T in gt]
        gt_rot = [matrix_to_quaternion(T[:3, :3]) for T in gt]
        gt = [gt_pos, gt_rot]
        #print(summary(model, torch.zeros(2, 2, 3, 6, 64, 1800)))

        outputs = model([images, imu])
        loss = simple_geo_const_loss(outputs, gt)
        print(outputs)


if __name__ == '__main__':
    from pytorch_model_summary import summary

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose([transfromers.ToTensor()])

    dataset = kitti.Kitti(config=cfg, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = deeplio_nets.DeepLIOS3(6, p=0)

    for epoch in range(0, 1):
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, epoch)
        print('-------------------------------------------------------------------')
        print()





