import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_net import BaseNet, eval_output_size_detection, num_flat_features
from .modules import Fire, SELayer
from .pointseg_net import PointSegNet


class DeepLIO0(BaseNet):
    """
    DeepLIO with Simple Siamese SqueezeNet
    """
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(DeepLIO0, self).__init__(input_shape, cfg)

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
        self.fire12 = nn.Sequential(Fire(2*feat_out_c, 64, 256, 256, bn=True, bn_d=bn_d),
                                    Fire(512, 64, 256, 256, bn=True, bn_d=bn_d),
                                    SELayer(512, reduction=2),
                                    nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)))

        self.fire34 = nn.Sequential(Fire(512, 80, 384, 384, bn=True, bn_d=bn_d),
                                    Fire(768, 80, 384, 384, bn=True, bn_d=bn_d),
                                    #nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)),
                                    nn.AdaptiveAvgPool2d((1, 1)))

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


class FeatureNetSimple0(BaseNet):
    def __init__(self, input_shape, cfg):
        super(FeatureNetSimple0, self).__init__(input_shape, cfg)

        self.conv1 = nn.Conv2d(self.c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv4 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv5 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.pool1(out)

        out = F.relu(self.conv2(out))
        out = self.bn2(out)
        out = self.pool2(out)

        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        out = self.pool3(out)

        out = F.relu(self.conv4(out))
        out = self.bn4(out)
        out = self.pool4(out)

        out = F.relu(self.conv5(out))
        out = self.bn5(out)
        out = self.pool5(out)
        return out


class FeatureNetSimple1(BaseNet):
    def __init__(self, input_shape, cfg):
        super(FeatureNetSimple1, self).__init__(input_shape, cfg)

        self.bypass = self.cfg.get('bypass', False)

        self.conv1 = nn.Conv2d(self.c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv5 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv7 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.pool7 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

    def forward(self, x):
        # 1. block
        out = F.relu(self.conv1(x), inplace=True)
        out = self.bn1(out)
        out = self.pool1(out)

        # 2. block
        out = F.relu(self.conv2(out), inplace=True)
        out = self.bn2(out)
        out = self.pool2(out)

        # 3. block
        out = F.relu(self.conv3(out), inplace=True)
        out = self.bn3(out)
        identitiy = out

        out = F.relu(self.conv4(out), inplace=True)
        out = self.bn4(out)
        if self.bypass:
            out += identitiy
        out = self.pool4(out)

        # 4. block
        out = F.relu(self.conv5(out), inplace=True)
        out = self.bn5(out)
        identitiy = out

        out = F.relu(self.conv6(out), inplace=True)
        out = self.bn6(out)
        if self.bypass:
            out += identitiy
        out = self.pool6(out)

        out = F.relu(self.conv7(out), inplace=True)
        out = self.bn7(out)
        out = self.pool7(out)
        return out


class DeepLIOS0N(BaseNet):
    """
        DeepLIO with simple siamese conv layers
        """
    def __init__(self, input_shape, cfg):
        super(DeepLIOS0N, self).__init__(input_shape, cfg)

    def create_network(self):
        cfg = self.cfg['feature-net']
        net_name = cfg.get('name', 'simple0')
        if net_name == 'simple0':
            self.feature_net = FeatureNetSimple0((self.h, self.w, self.c), cfg)
        elif net_name == 'simple1':
            self.feature_net = FeatureNetSimple1((self.h, self.w, self.c), cfg)
        else:
            raise ValueError("Featurenetwok {} is not supported!".format(net_name))

        # in-feature size autodetection
        self.feature_net.eval()
        with torch.no_grad():
            x = torch.randn((1, self.c, self.h, self.w))
            x = self.feature_net(x)
            _, c, h, w = x.shape
            self.fc_in_shape = (c, h, w)

        self.fc1 = nn.Linear(1*c*h*w, 512)
        self.fc2 = nn.Linear(512, 256)

        if self.p > 0:
            self.dropout = nn.Dropout2d(p=self.p)

        self.fc_pos = nn.Linear(256, 3)
        self.fc_ori = nn.Linear(256, 4)

    def forward(self, x):
        out = self.forward_feat_net(x)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        if self.p > 0:
            out = self.dropout(out)

        pos = self.fc_pos(out)
        ori = self.fc_ori(out)
        return pos, ori, None, None

    def forward_feat_net(self, x):
        raise NotImplementedError()


class DeepLIOS0(DeepLIOS0N):
    """
    DeepLIO with simple siamese conv layers
    """
    def __init__(self, input_shape, cfg):
        super(DeepLIOS0, self).__init__(input_shape, cfg)
        self.create_network()

    def forward_feat_net(self, x):
        images = x

        imgs_0 = images[0]
        imgs_1 = images[1]

        out_0 = self.feature_net(imgs_0)
        out_1 = self.feature_net(imgs_1)
        out = torch.sub(out_1, out_0) # torch.cat((out_1, out_0), dim=1)
        out = out.view(-1, num_flat_features(out))
        return out


class DeepLIOS1(DeepLIOS0N):
    """
    DeepLIO with simple siamese conv layers
    """
    def __init__(self, input_shape, cfg):
        super(DeepLIOS1, self).__init__(input_shape, cfg)

        self.c *= 2
        self.create_network()

    def forward_feat_net(self, x):
        images = x
        imgs0 = images[0]
        imgs1 = images[1]
        imgs = torch.stack(list(map(torch.cat, zip(imgs0, imgs1))))

        out = self.feature_net(imgs)
        out = out.view(-1, num_flat_features(out))
        return out