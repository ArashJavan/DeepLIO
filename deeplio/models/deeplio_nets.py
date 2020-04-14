import torch
import torch.nn as nn
import torch.nn.functional as F

from deeplio.models.squeeze_net import *


class DeepLIOS3(nn.Module):
    """
    DeepLIO with Simple Siamese SqueezeNet
    """
    def __init__(self, input_shape, p=0.):
        super(DeepLIOS3, self).__init__()
        self.p = p

        c, h, w = input_shape # number of channels, width and height
        # Siamese sqeeuze feature extraction networks
        self.feat_net1 = SqueezeNet(c, p=p)
        self.feat_net2 = SqueezeNet(c, p=p)

        # Odometry network
        self.fire1 = Fire(2 * 512, 64, 256, 256)
        self.fire2 = Fire(512, 64, 256, 256)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 0), ceil_mode=True)

        self.fire3 = Fire(512, 80, 256, 256)
        self.fire4 = Fire(512, 80, 256, 256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 0), ceil_mode=True)

        self.fire5 = Fire(512, 80, 384, 384)
        self.fire6 = Fire(768, 80, 384, 384)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 0), ceil_mode=True)

        self.fire7 = Fire(768, 80, 384, 384)
        self.fire8 = Fire(768, 80, 384, 384)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 0), ceil_mode=True)

        self.fc1 = nn.Linear(768 * 5 * 3, 256)
        self.fc2 = nn.Linear(256, 124)

        if self.p > 0:
            self.fropout = nn.Dropout2d(p=self.p)

        self.fc_pos = nn.Linear(124, 3)
        self.fc_ori = nn.Linear(124, 4)

    def forward(self, x):
        images = x

        imgs_0 = images[0]
        imgs_1 = images[1]

        feat_0 = self.feat_net1(imgs_0)
        feat_1 = self.feat_net2(imgs_1)

        feats = torch.cat((feat_0, feat_1), dim=1)

        out = self.fire1(feats)
        out = self.fire2(out)
        out = self.pool1(out)

        out = self.fire3(out)
        out = self.fire4(out)
        out = self.pool2(out)

        out = self.fire5(out)
        out = self.fire6(out)
        out = self.pool3(out)

        out = self.fire7(out)
        out = self.fire8(out)
        out = self.pool4(out)

        a, b, c, d = out.size()
        out = out.view(-1, b * c * d)

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        pos = self.fc_pos(out)
        ori = self.fc_ori(out)
        return [pos, ori]


class DeepLIOS0(nn.Module):
    """
    DeepLIO with simple siamese conv layers
    """
    def __init__(self, input_shape, p=0.):
        super(DeepLIOS0, self).__init__()
        self.p = p

        c, h, w = input_shape # number of channels, width and height
        # Siamese sqeeuze feature extraction networks

        self.siamese_net = self.create_inner_net(channels=c)

        # in-feature size autodetection
        x = torch.randn((1, c, h, w))
        x = self.siamese_net(x).detach()
        self.siamese_net.zero_grad()
        _, c, h, w = x.shape
        self.fc_in_shape = (c, h, w)

        self.fc1 = nn.Linear(c*h*w, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)

        self.fc_pos = nn.Linear(16, 3)
        #self.fc_ori = nn.Linear(64, 4)

    def create_inner_net(self, channels):
        net = nn.Sequential(
                nn.Conv2d(channels, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),

                nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1),
                nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True),
                nn.ReLU(True),

                nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),

                nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),
                nn.ReLU(True),

                nn.Conv2d(64, out_channels=128, kernel_size=3, stride=(1, 1), padding=1),
                nn.ReLU(True),

                nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),
                nn.ReLU(True),

                nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1),
                nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True),
                nn.ReLU(True),
            )
        return net

    def forward(self, x):
        images = x

        imgs_0 = images[0]
        imgs_1 = images[1]

        # outs = []
        # length = imgs_0.shape[0]
        # for i in range(length):
        #     im0, im1 = imgs_0[i:i+1], imgs_1[i:i+1]
        #     res0 = self.siamese_net(im0)
        #     res1 = self.siamese_net(im1)
        #     res0 = res0.view(-1, self.num_flat_features(res0))
        #     res1 = res1.view(-1, self.num_flat_features(res1))
        #     outs.append(res0 - res1)
        # out = torch.stack(outs).squeeze()

        out_0 = self.siamese_net(imgs_0)
        out_1 = self.siamese_net(imgs_1)
        out_0 = out_0.view(-1, self.num_flat_features(out_0))
        out_1 = out_1.view(-1, self.num_flat_features(out_1))
        out = torch.abs(out_1 - out_0)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        pos = self.fc_pos(out)
        #ori = torch.zeros((1, 3), requires_grad=False) #self.fc_ori(out)
        return pos

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def repr(self):
        return "{}\nIn-features 1st Linear: {}".format(self.__class__.__name__, self.fc_in_shape)


















