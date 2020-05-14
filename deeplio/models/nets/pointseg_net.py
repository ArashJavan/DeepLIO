import os
import datetime
import copy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Fire, FireDeconv, SELayer, ASPP


class PointSegNet(nn.Module):
    def __init__(self, input_shape, cfg):
        super(PointSegNet, self).__init__()
        bn_d = 0.1
        num_classes = len(cfg['classes'])
        self.p = cfg['dropout']
        self.input_shape = input_shape

        h, w, c = self.input_shape

        # Ecnoder part
        self.conv1a = nn.Sequential(nn.Conv2d(c, 64, kernel_size=3, stride=(1, 2), padding=1),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        self.conv1b = nn.Sequential(nn.Conv2d(c, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64, momentum=bn_d),
                                    nn.ReLU(inplace=True))

        self.fire23 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),
                                    Fire(64, 16, 64, 64, bn=True, bn_d=bn_d),
                                    Fire(128, 16, 64, 64, bn=True, bn_d=bn_d),
                                    SELayer(128, reduction=2))

        self.fire45 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),
                                    Fire(128, 32, 128, 128, bn=True, bn_d=bn_d),
                                    Fire(256, 32, 128, 128, bn=True, bn_d=bn_d),
                                    SELayer(256, reduction=2))

        self.fire6789 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=1),
                                      Fire(256, 48, 192, 192, bn=True, bn_d=bn_d),
                                      Fire(384, 48, 192, 192, bn=True, bn_d=bn_d),
                                      Fire(384, 64, 256, 256, bn=True, bn_d=bn_d),
                                      Fire(512, 64, 256, 256, bn=True, bn_d=bn_d),
                                      SELayer(512, reduction=2))

        self.aspp = ASPP(512, [6, 9, 12])

        # Decoder part
        self.fdeconv_el = FireDeconv(128, 32, 128, 128, bn=True, bn_d=bn_d)

        self.fdeconv_1 = FireDeconv(512, 64, 128, 128, bn=True, bn_d=bn_d)
        self.fdeconv_2 = FireDeconv(512, 64, 64, 64, bn=True, bn_d=bn_d)
        self.fdeconv_3 = FireDeconv(128, 16, 32, 32, bn=True, bn_d=bn_d)
        self.fdeconv_4 = FireDeconv(64, 16, 32, 32, bn=True, bn_d=bn_d)

        self.drop = nn.Dropout2d(p=self.p)
        self.conv2 = nn.Sequential(nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_1a = self.conv1a(x) # (H, W/2)
        x_1b = self.conv1b(x)

        ### Encoder forward
        # first fire block
        x_f23 = self.fire23(x_1a)

        # second fire block
        x_f45 = self.fire45(x_f23)

        # third fire block
        x_f6789 = self.fire6789(x_f45)

        # EL forward
        x_el = self.aspp(x_f6789)
        x_el = self.fdeconv_el(x_el)

        ### Decoder forward
        x_fd1 = self.fdeconv_1(x_f6789)  # (H, W/8)
        x_fd1_fused = torch.add(x_fd1, x_f45)
        x_fd1_fused = torch.cat((x_fd1_fused, x_el), dim=1)

        x_fd2 = self.fdeconv_2(x_fd1_fused)  # (H, W/4)
        x_fd2_fused = torch.add(x_fd2, x_f23)

        x_fd3 = self.fdeconv_3(x_fd2_fused)  # (H, W/2)
        x_fd3_fused = torch.add(x_fd3, x_1a)

        x_fd4 = self.fdeconv_4(x_fd3_fused)  # (H, W/2)
        x_fd4_fused = torch.add(x_fd4, x_1b)

        x_d = self.drop(x_fd4_fused)
        x = self.conv2(x_d)

        x_feat = x_f6789
        return x, x_feat


feat_names = [
    "x_in",
    "x_1b",
    "x_1a",
    "x_f23",
    "x_f45",
    "x_f6789",
    "x_el",
    "x_fd1",
    "x_fd1_fused",
    "x_fd2",
    "x_fd3",
    "x_fd4",
    "x_d",
    "x"]

index = 0
dname = os.path.abspath(os.path.dirname(__file__))
content_dir = os.path.abspath("{}/..".format(dname))
plt.ioff()
def plot_module(feats):
    global index

    index += 1
    if index % 20 != 0:
        return

    img_path = Path("{}/images".format(content_dir))
    img_path.mkdir(parents=True, exist_ok=True)

    # output
    x_lst = np.argmax(feats[-1], axis=0)
    x8c = np.zeros((x_lst.shape[0], x_lst.shape[1], 3))
    colors = [[0., 0., 0.],
              [0.9, 0., 0.],
              [0., 0.9, 0.],
              [0., 0., 0.9]]
    for j in range(4):
        x8c[x_lst == j, :] = colors[j]

    fig, ax = plt.subplots(2, 7, figsize=(20, 10))
    axes = ax.flatten()
    for i in range(len(feats)):
        x = feats[i]
        min_ = x.min()
        max_ = x.max()
        if i < (len(feats) - 1):
            x = (x - min_) / (max_ - min_ + 1e-6)
            axes[i].imshow(x)
            axes[i].set_title("{}, {:.5f}-{:.5f}".format(feat_names[i], min_, max_))
        else:
            axes[i].imshow(x8c)
            axes[i].set_title("{}, {:.5f}-{:.5f}".format(feat_names[i], min_, max_))

    im_path = "{}/{}_{}.png".format(str(img_path), index, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    fig.savefig(im_path, bbox_inches='tight')
    plt.close(fig)

