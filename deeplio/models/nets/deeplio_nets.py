import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_net import BaseNet, eval_output_size_detection, num_flat_features
from .modules import Fire, SELayer
from .pointseg_net import PointSegNet


class DeepLIOS3(BaseNet):
    """
    DeepLIO with Simple Siamese SqueezeNet
    """
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(DeepLIOS3, self).__init__(input_shape, cfg)

        # Siamese sqeeuze feature extraction networks
        feat_cfg = cfg['feature-net']
        self.feat_net = PointSegNet(input_shape, feat_cfg)
        if feat_cfg['pretrained']:
            ckp_path = feat_cfg['model-path']
            if os.path.isfile(ckp_path):
                print("loading pointseg checkpoint {}".format(ckp_path))
                checkpoint = torch.load(ckp_path)
                self.feat_net.load_state_dict(checkpoint['state_dict'])

        # Output size detection of feature extraction layer
        self.feat_net.eval()
        with torch.no_grad():
            x = torch.randn((1, self.c, self.h, self.w))
            x_feat = self.feat_net(x)[1]
            _, feat_out_c, feat_out_h, feat_out_w = x_feat.shape

        # odometry network
        self.fire12 = nn.Sequential(Fire(2*feat_out_c, 64, 64, 64, bn=True, bn_d=bn_d),
                                    Fire(128, 64, 64, 64, bn=True, bn_d=bn_d),
                                    SELayer(128, reduction=2),
                                    nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)))

        self.fire34 = nn.Sequential(Fire(128, 80, 64, 64, bn=True, bn_d=bn_d),
                                    Fire(128, 80, 64, 64, bn=True, bn_d=bn_d),
                                    #SELayer(512, reduction=2),
                                    nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)))

        # output middle fire conv-layers
        x = torch.rand((1, 2*feat_out_c, feat_out_h, feat_out_w))
        self.eval()
        with torch.no_grad():
            x = self.fire12(x)
            x = self.fire34(x)
            x = x.view(-1, num_flat_features(x))
        _, mid_c = x.shape

        self.fc1 = nn.Linear(mid_c, 512)

        if self.p > 0:
            self.dropout = nn.Dropout2d(p=self.p)

        self.fc_pos = nn.Linear(512, 3)
        self.fc_ori = nn.Linear(512, 4)

    def forward(self, x):
        x0 = x[0]
        x1 = x[1]

        x0_mask, x0_feat = self.feat_net(x0)
        x1_mask, x1_feat = self.feat_net(x1)

        x = torch.cat((x0_feat, x1_feat), dim=1)
        x = self.fire12(x)
        x = self.fire34(x)

        x = x.view(-1, num_flat_features(x))
        x = self.fc1(x)

        if self.p > 0:
            x = self.dropout(x)

        x_pos = self.fc_pos(x)
        x_ori = self.fc_ori(x)

        return x_pos, x_ori, x0_mask, x1_mask


class DeepLIOS0(BaseNet):
    """
    DeepLIO with simple siamese conv layers
    """
    def __init__(self, input_shape, cfg):
        super(DeepLIOS0, self).__init__(input_shape, cfg)

        # Siamese sqeeuze feature extraction networks
        self.siamese_net1 = self.create_inner_net(channels=self.c)
        #self.siamese_net2 = self.create_inner_net(channels=self.c)

        # in-feature size autodetection
        self.siamese_net1.eval()
        with torch.no_grad():
            x = torch.randn((1, self.c, self.h, self.w))
            x = self.siamese_net1(x)
            _, c, h, w = x.shape
            self.fc_in_shape = (c, h, w)

        self.fc1 = nn.Linear(1*c*h*w, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)

        if self.p > 0:
            self.dropout = nn.Dropout2d(p=self.p)

        self.fc_pos = nn.Linear(32, 3)
        self.fc_ori = nn.Linear(32, 4)

    def create_inner_net(self, channels):
        net = nn.Sequential(
                nn.Conv2d(channels, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),
        )
        return net

    def forward(self, x):
        images = x

        imgs_0 = images[0]
        imgs_1 = images[1]

        out_0 = self.siamese_net1(imgs_0)
        out_1 = self.siamese_net1(imgs_1)
        out = torch.sub(out_1, out_0) # torch.cat((out_1, out_0), dim=1)
        out = out.view(-1, num_flat_features(out))

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        if self.p > 0:
            out = self.dropout(out)

        pos = self.fc_pos(out)
        ori = self.fc_ori(out)
        return pos, ori, None, None


class DeepLIOS01(BaseNet):
    """
    DeepLIO with simple siamese conv layers
    """
    def __init__(self, input_shape, cfg):
        super(DeepLIOS01, self).__init__(input_shape, cfg)

        self.c *= 2

        # Siamese sqeeuze feature extraction networks
        self.siamese_net = self.create_inner_net(channels=self.c)

        # in-feature size autodetection
        self.siamese_net.eval()
        with torch.no_grad():
            x = torch.randn((1, self.c, self.h, self.w))
            x = self.siamese_net(x)
            _, c, h, w = x.shape
            self.fc_in_shape = (c, h, w)

        self.fc1 = nn.Linear(c*h*w, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)

        if self.p > 0:
            self.dropout = nn.Dropout2d(p=self.p)

        self.fc_pos = nn.Linear(32, 3)
        self.fc_ori = nn.Linear(32, 4)

    def create_inner_net(self, channels):
        net = nn.Sequential(
                nn.Conv2d(channels, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),
        )
        return net

    def forward(self, x):
        images = x
        imgs0 = images[0]
        imgs1 = images[1]
        imgs = torch.stack(list(map(torch.cat, zip(imgs0, imgs1))))

        out = self.siamese_net(imgs)
        out = out.view(-1, num_flat_features(out))

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        if self.p > 0:
            out = self.dropout(out)

        pos = self.fc_pos(out)
        ori = self.fc_ori(out)
        return pos, ori, None, None


















