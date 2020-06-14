import torch
from torch import nn
from torch.nn import functional as F

from .base_net import BaseNet, num_flat_features
from .pointseg_modules import Fire, SELayer
from .pointseg_net import PSEncoder, PSDecoder
from ..misc import get_config_container


class BaseLidarFeatNet(BaseNet):
    def __init__(self, input_shape, cfg):
        super(BaseLidarFeatNet, self).__init__()
        self.input_shape = input_shape
        self.p = cfg['dropout']
        self.fusion = cfg['fusion']
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations
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
        input = torch.rand((1, self.seq_size+1, c, h, w))
        self.eval()
        with torch.no_grad():
            out = self.forward(input)
        return out.shape

    def get_output_shape(self):
        return self.output_shape


class LidarPointSegFeat(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(LidarPointSegFeat, self).__init__(input_shape, cfg)
        self.part = cfg['part'].lower()
        self.bn_d = bn_d

        self.encoder = PSEncoder(input_shape, cfg)

        # shapes of  x_1a, x_1b, x_se1, x_se2, x_se3, x_el
        enc_out_shapes = self.encoder.get_output_shape()
        if self.part == 'encoder+decoder':
            self.decoder = PSDecoder(enc_out_shapes, cfg)
            dec_out_shape = self.decoder.get_output_shape()

        # number of output channels in encoder
        b, c, h, w = enc_out_shapes[4]

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

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """

        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        batch_size = x.shape[0]

        x_0, x_1 = self.combine_data(x)

        x_mask_0 = torch.zeros((1), requires_grad=False)
        x_mask_1 = torch.zeros((1), requires_grad=False)

        x_1a_0, x_1b_0, x_se1_0, x_se2_0, x_se3_0, x_el_0 = self.encoder(x_0)
        if self.part == 'encoder+decoder':
            x_mask_0 = self.decoder([x_1a_0, x_1b_0, x_se1_0, x_se2_0, x_se3_0, x_el_0])
        x_feat_0 = x_se3_0

        x_1a_1, x_1b_1, x_se1_1, x_se2_1, x_se3_1, x_el_1 = self.encoder(x_1)
        if self.part == 'encoder+decoder':
            x_mask_1 = self.decoder([x_1a_1, x_1b_1, x_se1_1, x_se2_1, x_se3_1, x_el_1])
        x_feat_1 = x_se3_1

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        else:
            x = torch.sub(x_feat_0, x_feat_1)

        x = self.fire12(x)
        x = self.fire34(x)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(batch_size, self.seq_size, num_flat_features(x, 1))
        return x


class LidarSimpleFeat0(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super(LidarSimpleFeat0, self).__init__(input_shape, cfg)
        self.encoder = FeatureNetSimple0(self.input_shape)

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        batch_size = x.shape[0]
        x_0, x_1 = self.combine_data(x)

        x_feat_0 = self.encoder(x_0)
        x_feat_1 = self.encoder(x_1)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        else:
            x = torch.sub(x_feat_0, x_feat_1)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(batch_size, self.seq_size, num_flat_features(x, 1))
        return x


class LidarSimpleFeat1(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super(LidarSimpleFeat1, self).__init__(input_shape, cfg)
        bypass = cfg['bypass']
        self.encoder = FeatureNetSimple1(self.input_shape, bypass=bypass)

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param inputs: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return: outputs: features of dim [BxTxN]
        mask0: predicted mask to each time sequence
        """
        batch_size = x.shape[0]
        x_0, x_1 = self.combine_data(x)

        x_feat_0 = self.encoder(x_0)
        x_feat_1 = self.encoder(x_1)

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        else:
            x = torch.sub(x_feat_0, x_feat_1)

        if self.p > 0.:
            x = self.drop(x)

        # reshape output to BxTxCxHxW
        x = x.view(batch_size, self.seq_size, num_flat_features(x, 1))
        return x


class FeatureNetSimple0(nn.Module):
    """Simple Conv. based Feature Network
    """
    def __init__(self, input_shape):
        super(FeatureNetSimple0, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = nn.Conv2d(c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
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


class FeatureNetSimple1(nn.Module):
    """Simple Conv. based Feature Network with optinal bypass connections"""
    def __init__(self, input_shape, bypass=False):
        super(FeatureNetSimple1, self).__init__()

        self.bypass = bypass
        self.input_shape = input_shape
        c, h, w = self.input_shape

        self.conv1 = nn.Conv2d(c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
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

        self.conv5 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)

        self.conv7 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
        self.bn7 = nn.BatchNorm2d(128)
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

