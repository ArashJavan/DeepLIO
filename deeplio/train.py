import os
import yaml
import argparse
import sys
import datetime

import scipy

import torch
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

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = GeoConstLoss()

    running_loss = 0.
    for idx, data in enumerate(train_loader):

        # skip unvalid data without ground-truth
        if not torch.all(data['valid']):
            continue

        imgs_0 = data['images'][0][0]
        imgs_1 = data['images'][1][0]
        images = [imgs_0.to(device), imgs_1.to(device)]
        # plot_images(images)

        gt = [data['ground-truth'][i][0][-1] for i in range(len(data['ground-truth']))]
        gt_pos = np.array([T[:3, 3].numpy() for T in gt])
        gt_rot =  np.array([matrix_to_quaternion(T[:3, :3]) for T in gt])
        gt = [torch.from_numpy(gt_pos).to(device), torch.from_numpy(gt_rot).to(device)]
        #print(summary(model, torch.zeros(2, 3, 2, 64, 1024).to(device)))
        #print("combinations: {}".format(data['combinations']))

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, gt)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 10 == 0:
            writer.add_scalar("Loss/train", running_loss / 10, len(train_loader) + idx)
            running_loss = 0.

            comb = data['combinations'][0].numpy()
            len_seq = len(np.unique(comb))
            images = []
            n_channels = len(cfg['channels'])
            ims = [imgs_0, imgs_1]
            for u in range(len_seq):
                for v in range(len(comb)):
                    if u == comb[v, 0]:
                        images.append(ims[0][v])
                        break
                    elif u == comb[v, 1]:
                        images.append(ims[1][v])
                        break
            images = torch.stack(images)
            n, c, h, w = images.shape
            #for ch in range(n_channels):
            #    writer.add_image("Images/Channel-{}".format(u), images[:, u, :, :].reshape(n, 1, h, w), dataformats='NCHW')

            print("[{}] loss: {}".format(idx, loss.data))
            preds_ = [outputs[i].cpu().detach().numpy() for i in range(2)]
            gts_ = [gt[i].cpu().detach().numpy() for i in range(2)]
            print("{}\n{}".format(preds_[0], gts_[0]))


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
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    channels = len(cfg['channels'])
    model = deeplio_nets.DeepLIOS0(input_shape=(channels, None, None), p=0)
    model.to(device)

    for epoch in range(0, 1):
        print('-------------------------------------------------------------------')
        train(model, train_dataloader, epoch, writer)
        print('-------------------------------------------------------------------')
        print()





