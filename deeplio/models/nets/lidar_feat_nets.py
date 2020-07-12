import torch
from torch import nn
from torch.nn import functional as F

from .base_net import BaseNet, num_flat_features, conv
from .pointseg_modules import Fire, SELayer
from .pointseg_net import PSEncoder2
from .resnet import ResNetEncoder
from ..misc import get_config_container


class BaseLidarFeatNet(BaseNet):
    def __init__(self, input_shape, cfg):
        super(BaseLidarFeatNet, self).__init__()
        self.p = cfg['dropout']
        self.fusion = cfg['fusion']
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.timestamps = self.cfg_container.timestamps
        self.combinations = self.cfg_container.combinations
        self.input_shape = input_shape
        self.output_shape = None

    def combine_data(self, x):
        b, s, c, h, w = x.shape
        comb_idx_0 = self.combinations[:, 0]
        comb_idx_1 = self.combinations[:, 1]

        ims0 = x[:, comb_idx_0].view(b*self.seq_size, c, h, w).contiguous()
        ims1 = x[:, comb_idx_1].view(b*self.seq_size, c, h, w).contiguous()
        return ims0, ims1

    def calc_output_shape(self):
        c, h, w = self.input_shape
        input1 = torch.rand((1, self.seq_size, self.timestamps, c, h, w))
        input2 = torch.rand((1, self.seq_size, self.timestamps, c, h, w))
        self.eval()
        with torch.no_grad():
            out = self.forward([input1, input2])
        return out.shape

    def get_output_shape(self):
        return self.output_shape


class LidarPointSegFeat(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(LidarPointSegFeat, self).__init__(input_shape, cfg)
        self.part = cfg['part'].lower()
        self.bn_d = bn_d

        c, h, w = self.input_shape

        self.encoder1 = PSEncoder2((2*c, h, w), cfg)
        self.encoder2 = PSEncoder2((2*c, h, w), cfg)

        # shapes of  x_1a, x_1b, x_se1, x_se2, x_se3, x_el
        enc_out_shapes = self.encoder1.get_output_shape()

        # number of output channels in encoder
        b, c, h, w = enc_out_shapes

        alpha = 2 if self.fusion == 'cat' else 1
        self.fire12 = nn.Sequential(Fire(alpha*c, 64, 256, 256, bn=True, bn_d=self.bn_d),
                                    Fire(512, 64, 256, 256, bn=True, bn_d=self.bn_d),
                                    SELayer(512, reduction=2),
                                    nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)))

        self.fire34 = nn.Sequential(Fire(512, 80, 384, 384, bn=True, bn_d=self.bn_d),
                                    Fire(768, 80, 384, 384, bn=True, bn_d=self.bn_d),
                                    #nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)),
                                    nn.AdaptiveAvgPool2d((1, 1)))

        if self.p > 0.:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(768, 128)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """

        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[0], x[1]
        b, s, t, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b * s, t * c, h, w)
        imgs_normals = imgs_xyz.reshape(b * s, t * c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1
        x = x[:, :, 0, 0]
        #x = self.fire12(x)
        #x = self.fire34(x)[:, :, 0, 0]

        if self.p > 0.:
            x = self.drop(x)

        x = F.leaky_relu(self.fc1(x))

        # reshape output to BxTxCxHxW
        x = x.view(b, s, num_flat_features(x, 1))
        return x


class LidarFlowNetFeat(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super(LidarFlowNetFeat, self).__init__(input_shape, cfg)
        c, h, w = self.input_shape
        batch_norm = True

        self.encoder1 = FlowNetEncoder([2*c, h, w])
        self.encoder2 = FlowNetEncoder([2*c, h, w])

        self.conv1 = conv(batch_norm, 512, 512, stride=2)
        self.conv2 = conv(batch_norm, 512, 512, stride=2)
        self.conv3 = conv(batch_norm, 512, 1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[0], x[1]

        b, s, t, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b * s, t * c, h, w)
        imgs_normals = imgs_xyz.reshape(b * s, t * c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(b, s, num_flat_features(x, 1))
        return x


class LidarResNetFeat(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super(LidarResNetFeat, self).__init__(input_shape, cfg)
        c, h, w = self.input_shape
        self.encoder1 = ResNetEncoder([2*c, h, w])
        self.encoder2 = ResNetEncoder([2*c, h, w])

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(512, 128)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        imgs_xyz, imgs_normals = x[0], x[1]

        b, s, t, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b * s, t * c, h, w)
        imgs_normals = imgs_xyz.reshape(b * s, t * c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        if self.p > 0.:
            x = self.drop(x)

        x = F.leaky_relu(self.fc1(x))

        # reshape output to BxTxCxHxW
        x = x.view(b, s, num_flat_features(x, 1))
        return x


class LidarSimpleFeat1(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super(LidarSimpleFeat1, self).__init__(input_shape, cfg)
        bypass = cfg['bypass']
        c, h, w = self.input_shape

        self.encoder1 = FeatureNetSimple1([2*c, h, w], bypass=bypass)
        self.encoder2 = FeatureNetSimple1([2*c, h, w], bypass=bypass)

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(512, 128)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxSxTxCxHxW], S:=Seq-length T:=#timestamps, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        imgs_xyz, imgs_normals = x[0], x[1]

        b, s, t, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b * s, t * c, h, w)
        imgs_normals = imgs_xyz.reshape(b * s, t * c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == 'cat':
            y = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == 'add':
            y = x_feat_0 + x_feat_1
        else:
            y = x_feat_0 - x_feat_1

        if self.p > 0.:
            y = self.drop(y)

        y = F.leaky_relu(self.fc1(y[:, :, 0, 0]))

        # reshape output to BxTxCxHxW
        y = y.view(b, s, num_flat_features(y, 1))
        return y


class FlowNetEncoder(nn.Module):
    """Simple Conv. based Feature Network
    """
    def __init__(self, input_shape, batch_norm=True):
        super(FlowNetEncoder, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = conv(batch_norm, c, 64, kernel_size=(5, 7), stride=(1, 2))
        self.conv2 = conv(batch_norm, 64, 128, kernel_size=(3, 5), stride=(1, 2))
        self.conv3 = conv(batch_norm, 128, 256, kernel_size=(3, 5), stride=(1, 2))
        self.conv3_1 = conv(batch_norm, 256, 256)
        self.conv4 = conv(batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(batch_norm, 512, 512)
        self.conv5 = conv(batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(batch_norm, 512, 512)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out = out_conv5
        return out


class FeatureNetSimple1(nn.Module):
    """Simple Conv. based Feature Network with optinal bypass connections"""
    def __init__(self, input_shape, bypass=False):
        super(FeatureNetSimple1, self).__init__()

        self.bypass = bypass
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = nn.Conv2d(c, out_channels=64, kernel_size=(5, 7), stride=(1, 2), padding=(2, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv2 = nn.Conv2d(64, out_channels=128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)

        self.conv3 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, out_channels=256, kernel_size=3, stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv5 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, out_channels=512, kernel_size=3, stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv7 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=(1, 1), padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        # self.pool7 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
        self.pool7 = nn.AdaptiveAvgPool2d((1, 1))

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

