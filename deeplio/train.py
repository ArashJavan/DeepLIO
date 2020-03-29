import os
import yaml
import argparse
import sys
import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from torch import optim
from torch.utils import tensorboard

from deeplio.datasets import kitti
from deeplio.models import deeplio_nets
from deeplio.datasets import transfromers
from deeplio.losses.losses import *
from deeplio.common.spatial import *
from deeplio.visualization.utilities import *

from pytorch_model_summary import summary

import matplotlib.pyplot as plt

def plot_images(images):
    img1, img2 = images[0], images[1]
    fig, ax = plt.subplots(3, 2)
    for i in range(1):
        for j in range(2):
            ax[i, j].imshow(images[j][i][0, :, :].cpu().numpy())
    fig.show()


def train(model, train_loader, epoch, writer):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = GeoConstLoss()

    running_loss = 0.
    for batch_index, data in enumerate(train_loader):

        # skip unvalid data without ground-truth
        if not data['valid']:
            continue

        images = [data['images'][0][0].to(device), data['images'][1][0].to(device)]
        # plot_images(images)

        gt = [data['ground-truth'][i][0][-1] for i in range(len(data['ground-truth']))]
        gt_pos = np.array([T[:3, 3].numpy() for T in gt])
        gt_rot =  np.array([matrix_to_quaternion(T[:3, :3]) for T in gt])
        gt = [torch.from_numpy(gt_pos).to(device), torch.from_numpy(gt_rot).to(device)]
        #print(summary(model, torch.zeros(2, 3, 2, 64, 1800).to(device)))
        #print("combinations: {}".format(data['combinations']))
        outputs = model(images)
        loss = criterion(outputs, gt)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_index % 10 == 0:
            writer.add_scalar("Loss/train", running_loss / 10, len(train_loader) + batch_index)
            running_loss = 0.
            print("loss: {}".format(loss.data))
            preds_ = [outputs[i].cpu().detach().numpy() for i in range(2)]
            gts_ = [gt[i].cpu().detach().numpy() for i in range(2)]
            print("{}\n{}\n{}\n{}".format(preds_[0], gts_[0], preds_[1], gts_[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLIO Training')

    parser.add_argument('--model_path', default='./model', type=str, help='path to where model')

    # Hyper Params
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optim')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay for optim')
    parser.add_argument('--lr_step', default=1000, type=int, help='number of lr step')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='gamma for lr scheduler')

    parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='number of epoch to start learning')
    parser.add_argument('--pretrain', default=False, type=bool, help='Whether or not to pretrain')
    parser.add_argument('--resume', default=False, type=bool, help='Whether or not to resume')

    # Device Option
    parser.add_argument('--gpu_ids', dest='gpu_ids', default=[0, 1], nargs="+", type=int, help='which gpu you use')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='mini-batch size')

    args = parser.parse_args()

    # To use TensorboardX
    writer = tensorboard.SummaryWriter()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("../config.yaml") as f:
        cfg = yaml.safe_load(f)

    mean = np.array(cfg['datasets']['kitti']['mean'])
    std = np.array(cfg['datasets']['kitti']['std'])

    transform = transforms.Compose([transfromers.ToTensor(),
                                    transfromers.Normalize(mean=mean, std=std)])

    dataset = kitti.Kitti(config=cfg, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    channels = len(cfg['channels'])
    model = deeplio_nets.DeepLIOS0(input_shape=(channels, None, None), p=0)
    model.to(device)

    for epoch in range(0, 1):
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, epoch, writer)
        print('-------------------------------------------------------------------')
        print()





