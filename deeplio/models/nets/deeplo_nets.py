import torch
from torch import nn
from torch.nn import functional as F

from deeplio.common.logger import get_app_logger
from .base_net import BaseDeepLIO, BaseNet, num_flat_features
from .modules import Fire, SELayer
from .pointseg_net import PSEncoder, PSDecoder
from ..misc import get_config_container

class DeepLO(BaseDeepLIO):
    """Base class for all DepplioN Networks"""
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(DeepLO, self).__init__()

        self.logger = get_app_logger()
        self.cfg = cfg['deeplo']
        self.p = self.cfg.get('dropout', 0.)
        self.bn_d = bn_d
        self.input_shape = input_shape

        self.feat_net = None
        self.odom_feat_net = None
        self.fc_pos = None
        self.fc_ori = None

    def initialize(self):
        if self.feat_net is None or self.odom_feat_net is None:
            raise ValueError("{}: feature networks are not defined!".format(self.name))

        in_features = self.odom_feat_net.get_output_shape()

        if self.p > 0:
            self.drop = nn.Dropout(self.p)
        self.fc_pos = nn.Linear(in_features, 3)
        self.fc_ori = nn.Linear(in_features, 4)

    def forward(self, x):
        x, x_mask_0, x_mask_1 = self.feat_net(x)
        x = self.odom_feat_net(x)

        if self.p > 0.:
            x = self.drop(x)

        x_pos = self.fc_pos(x)
        x_ori = self.fc_ori(x)
        return x_pos, x_ori

    @property
    def name(self):
        self_name = self.__class__.__name__.lower()
        if self.feat_net is None or self.odom_feat_net is None:
            res = "{}".format(self_name)
        else:
            res = "{}_{}_{}".format(self_name, self.feat_net.name,
                                 self.odom_feat_net.name)
        return res

    def get_feat_networks(self):
        return [self.feat_net, self.odom_feat_net]


class DeepLOPointSegFeat(BaseNet):
    def __init__(self, input_shape, cfg, bn_d=0.1):
        super(DeepLOPointSegFeat, self).__init__()
        self.input_shape = input_shape
        self.p = cfg['dropout']
        self.part = cfg['part'].lower()
        self.fusion = cfg['fusion']
        self.bn_d = bn_d
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations

        self.encoder = PSEncoder(input_shape, cfg)

        self.enc_out_shape = self.encoder.get_output_shape()
        if self.part == 'encoder+decoder':
            self.decoder = PSDecoder(self.enc_out_shape, cfg)
            dec_out_shape = self.decoder.get_output_shape()

        # number of output channels in encoder
        feat_out_c = self.enc_out_shape[0]

        alpha = 2 if self.fusion == 'cat' else 1
        self.fire12 = nn.Sequential(Fire(alpha*feat_out_c, 64, 256, 256, bn=True, bn_d=self.bn_d),
                                    Fire(512, 64, 256, 256, bn=True, bn_d=self.bn_d),
                                    SELayer(512, reduction=2),
                                    nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)))

        self.fire34 = nn.Sequential(Fire(512, 80, 384, 384, bn=True, bn_d=self.bn_d),
                                    Fire(768, 80, 384, 384, bn=True, bn_d=self.bn_d),
                                    #nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1)),
                                    nn.AdaptiveAvgPool2d((1, 1)))

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """

        :param x: images of dimension [BxTxCxHxW], where T is seq-size+1, e.g. 2+1
        :return:
        """
        x_0, x_1 = self.combine_data(x)

        x_mask_0 = torch.zeros((1), requires_grad=False)
        x_mask_1 = torch.zeros((1), requires_grad=False)

        x_1a_0, x_1b_0, x_se1_0, x_se2_0, x_se3_0, x_el_0 = self.encoder(x_0)
        if self.part == 'encoder+decoder':
            x_mask_0 = self.decoder(x_1a_0, x_1b_0, x_se1_0, x_se2_0, x_se3_0, x_el_0)
        x_feat_0 = x_se3_0

        x_1a_1, x_1b_1, x_se1_1, x_se2_1, x_se3_1, x_el_1 = self.encoder(x_1)
        if self.part == 'encoder+decoder':
            x_mask_1 = self.decoder(x_1a_1, x_1b_1, x_se1_1, x_se2_1, x_se3_1, x_el_1)
        x_feat_1 = x_se3_1

        if self.fusion == 'cat':
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        else:
            x = torch.sub(x_feat_0, x_feat_1)

        x = self.fire12(x)
        x = self.fire34(x)

        x = x.view(-1, num_flat_features(x))
        return x, x_mask_0, x_mask_1

    def combine_data(self, x):
        b, s, c, h, w = x.shape

        comb_idx_0 = self.combinations[:, 0]
        comb_idx_1 = self.combinations[:, 1]

        ims0 = x[:, comb_idx_0].view(b*self.seq_size, c, h, w)
        ims1 = x[:, comb_idx_1].view(b*self.seq_size, c, h, w)
        return ims0, ims1

    def calc_output_shape(self):
        c, h, w = self.input_shape
        input = torch.rand((1, self.seq_size+1, c, h, w))
        self.eval()
        with torch.no_grad():
            out, _, _ = self.forward(input)
        return out.shape

    def get_output_shape(self):
        return self.output_shape[1]


class DeepLOOdomFeatFC(BaseNet):
    def __init__(self, in_features, cfg):
        super(DeepLOOdomFeatFC, self).__init__()
        self.input_size = in_features
        self.hidden_size = cfg.get('hidden-size', [256, 128])
        num_layers = len(self.hidden_size)

        layers = [nn.Linear(self.input_size, self.hidden_size[0])]
        for i in range(1, num_layers):
            layers.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True)
        return x

    def get_output_shape(self):
        return self.hidden_size[-1]


class DeepLOOdomFeatRNN(BaseNet):
    def __init__(self, in_features, cfg):
        super(DeepLOOdomFeatRNN, self).__init__()
        rnn_type = cfg['type'].lower()
        num_layers = cfg.get('num-layers', 2)
        self.hidden_size = cfg.get('hidden-size', [6, 6])
        self.bidirectional = cfg.get('bidirectional', False)
        self.input_size = in_features

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size[0],
                              num_layers=num_layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0],
                               num_layers=num_layers, bidirectional=self.bidirectional)

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True)
        return x

    def get_output_shape(self):
        return self.hidden_size[-1]


#
#
# class DeepLIO0(DeepLIONBase):
#     """
#     DeepLIO with Simple Siamese SqueezeNet
#     """
#     def __init__(self, input_shape, cfg, bn_d=0.1):
#         super(DeepLIO0, self).__init__(input_shape, cfg)
#
#         mid_c = self.get_feature_channel_size()
#         self.fc1 = nn.Linear(mid_c, 512)
#
#         if self.p > 0:
#             self.dropout = nn.Dropout2d(p=self.p)
#
#         self.fc_pos = nn.Linear(512, 3)
#         self.fc_ori = nn.Linear(512, 4)
#
#     def forward(self, x):
#         x0 = x[0]
#         x1 = x[1]
#
#         x0_mask, x0_feat = self.feat_net(x0)
#         x1_mask, x1_feat = self.feat_net(x1)
#
#         x = torch.cat((x0_feat, x1_feat), dim=1)
#         x = self.fire12(x)
#         x = self.fire34(x)
#
#         x = x.view(-1, num_flat_features(x))
#         x = self.fc1(x)
#
#         if self.p > 0:
#             x = self.dropout(x)
#
#         x_pos = self.fc_pos(x)
#         x_ori = self.fc_ori(x)
#
#         return x_pos, x_ori, x0_mask, x1_mask
#
#
# class DeepLIO1(DeepLIONBase):
#     """
#     DeepLIO with Simple Siamese SqueezeNet
#     """
#     def __init__(self, input_shape, cfg, bn_d=0.1):
#         super(DeepLIO1, self).__init__(input_shape, cfg)
#
#         self.rnn_hidden_size = 6
#         self.rnn_num_layers = 1
#         self.bidrectional = False
#
#         self.rnn_imu = nn.LSTM(input_size=6, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers,
#                                batch_first=True, bidirectional=self.bidrectional)
#
#         mid_c = self.get_feature_channel_size()
#         self.fc1 = nn.Linear(mid_c + self.rnn_hidden_size, 512)
#
#         if self.p > 0:
#             self.dropout = nn.Dropout2d(p=self.p)
#
#         self.fc_pos = nn.Linear(512, 3)
#         self.fc_ori = nn.Linear(512, 4)
#
#     def forward(self, x):
#         ims0 = x[0]
#         ims1 = x[1]
#         imus = x[2]
#
#         n_batches, n_seq = len(imus), len(imus[0])
#         x_imu = []
#         for b in range(n_batches):
#             for s in range(n_seq):
#                 imu = imus[b][s]
#                 out, hidden = self.rnn_imu(imu.unsqueeze(0))
#                 x_imu.append(out[0, -1, :])
#
#         x_imu = torch.stack(x_imu)
#
#         x0_mask, x0_feat = self.feat_net(ims0)
#         x1_mask, x1_feat = self.feat_net(ims1)
#
#         x_feat = torch.cat((x0_feat, x1_feat), dim=1)
#         x_feat = self.fire12(x_feat)
#         x_feat = self.fire34(x_feat)
#
#         x_feat = x_feat.view(-1, num_flat_features(x_feat))
#         x = torch.cat((x_feat, x_imu), dim=1)
#         x = self.fc1(x)
#
#         if self.p > 0:
#             x = self.dropout(x)
#
#         x_pos = self.fc_pos(x)
#         x_ori = self.fc_ori(x)
#
#         return x_pos, x_ori, x0_mask, x1_mask
#
#
# class DeepLIOS0N(BaseNet):
#     """
#         DeepLIO Base class for simple siamese conv layers
#         """
#     def __init__(self, input_shape, cfg):
#         super(DeepLIOS0N, self).__init__(input_shape, cfg)
#
#     def create_network(self):
#         cfg = self.cfg['feature-net']
#         net_name = cfg.get('name', 'simple0')
#         if net_name == 'simple0':
#             self.feature_net = FeatureNetSimple0((self.h, self.w, self.c), cfg)
#         elif net_name == 'simple1':
#             self.feature_net = FeatureNetSimple1((self.h, self.w, self.c), cfg)
#         else:
#             raise ValueError("Featurenetwok {} is not supported!".format(net_name))
#
#         # in-feature size autodetection
#         self.feature_net.eval()
#         with torch.no_grad():
#             x = torch.randn((1, self.c, self.h, self.w))
#             x = self.feature_net(x)
#             _, c, h, w = x.shape
#             self.fc_in_shape = (c, h, w)
#
#         self.fc1 = nn.Linear(2*c*h*w, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#
#         if self.p > 0:
#             self.dropout = nn.Dropout2d(p=self.p)
#
#         self.fc_pos = nn.Linear(128, 3)
#         self.fc_ori = nn.Linear(128, 4)
#
#     def forward(self, x):
#         out = self.forward_feat_net(x)
#
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#
#         if self.p > 0:
#             out = self.dropout(out)
#
#         pos = self.fc_pos(out)
#         ori = self.fc_ori(out)
#         return pos, ori, None, None
#
#     def forward_feat_net(self, x):
#         raise NotImplementedError()
#
#
# class DeepLIOS0(DeepLIOS0N):
#     """
#     DeepLIOS0: In this netowrk the images in the forward pass are passed to the network as they
#     """
#     def __init__(self, input_shape, cfg):
#         super(DeepLIOS0, self).__init__(input_shape, cfg)
#         self.create_network()
#
#     def forward_feat_net(self, x):
#         images = x
#
#         imgs_0 = images[0]
#         imgs_1 = images[1]
#
#         out_0 = self.feature_net(imgs_0)
#         out_1 = self.feature_net(imgs_1)
#         out = torch.cat((out_1, out_0), dim=1) #  torch.sub(out_1, out_0) #
#         out = out.view(-1, num_flat_features(out))
#         return out
#
#
# class DeepLIOS1(DeepLIOS0N):
#     """
#     DeepLIOS1: In this netowrk the images in the forward pass are stacked together pariwise and the passed
#     to the network
#     """
#     def __init__(self, input_shape, cfg):
#         super(DeepLIOS1, self).__init__(input_shape, cfg)
#
#         self.c *= 2
#         self.create_network()
#
#     def forward_feat_net(self, x):
#         images = x
#         imgs0 = images[0]
#         imgs1 = images[1]
#         imgs = torch.stack(list(map(torch.cat, zip(imgs0, imgs1))))
#
#         out = self.feature_net(imgs)
#         out = out.view(-1, num_flat_features(out))
#         return out
#
#
# class FeatureNetSimple0(BaseNet):
#     """Simple Conv. based Feature Network
#     """
#     def __init__(self, input_shape, cfg):
#         super(FeatureNetSimple0, self).__init__(input_shape, cfg)
#
#         self.conv1 = nn.Conv2d(self.c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.pool3 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv4 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv5 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.pool5 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.bn1(out)
#         out = self.pool1(out)
#
#         out = F.relu(self.conv2(out))
#         out = self.bn2(out)
#         out = self.pool2(out)
#
#         out = F.relu(self.conv3(out))
#         out = self.bn3(out)
#         out = self.pool3(out)
#
#         out = F.relu(self.conv4(out))
#         out = self.bn4(out)
#         out = self.pool4(out)
#
#         out = F.relu(self.conv5(out))
#         out = self.bn5(out)
#         out = self.pool5(out)
#         return out
#
#
# class FeatureNetSimple1(BaseNet):
#     """Simple Conv. based Feature Network with optinal bypass connections"""
#     def __init__(self, input_shape, cfg):
#         super(FeatureNetSimple1, self).__init__(input_shape, cfg)
#
#         self.bypass = self.cfg.get('bypass', False)
#
#         self.conv1 = nn.Conv2d(self.c, out_channels=32, kernel_size=3, stride=(1, 2), padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv3 = nn.Conv2d(32, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#
#         self.conv4 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.pool4 = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv5 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn5 = nn.BatchNorm2d(128)
#
#         self.conv6 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn6 = nn.BatchNorm2d(128)
#         self.pool6 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
#
#         self.conv7 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=(1, 1), padding=1)
#         self.bn7 = nn.BatchNorm2d(128)
#         self.pool7 = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=(1, 1), ceil_mode=True)
#
#     def forward(self, x):
#         # 1. block
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.bn1(out)
#         out = self.pool1(out)
#
#         # 2. block
#         out = F.relu(self.conv2(out), inplace=True)
#         out = self.bn2(out)
#         out = self.pool2(out)
#
#         # 3. block
#         out = F.relu(self.conv3(out), inplace=True)
#         out = self.bn3(out)
#         identitiy = out
#
#         out = F.relu(self.conv4(out), inplace=True)
#         out = self.bn4(out)
#         if self.bypass:
#             out += identitiy
#         out = self.pool4(out)
#
#         # 4. block
#         out = F.relu(self.conv5(out), inplace=True)
#         out = self.bn5(out)
#         identitiy = out
#
#         out = F.relu(self.conv6(out), inplace=True)
#         out = self.bn6(out)
#         if self.bypass:
#             out += identitiy
#         out = self.pool6(out)
#
#         out = F.relu(self.conv7(out), inplace=True)
#         out = self.bn7(out)
#         out = self.pool7(out)
#         return out
