import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_net import BaseNet


class DeepLIONBase(BaseNet):
    def __init__(self, cfg):
        super(DeepLIONBase, self).__init__()
        self.cfg = cfg
        self.p = cfg['dropout']
        self.fusion_type = cfg.get('fusion', 'cat').lower()

        #self.input_shape = input_shape
        # number of channels, width and height
        #self.h, self.w, self.c = input_shape

        self.lidar_feat_net = None
        self.imu_feat_net = None
        self.odom_feat_net = None
        self.initilaize()

    def initilaize(self):
        if self.lidar_feat_net is None or self.imu_feat_net is None:
            raise ValueError("Both Lidar and IMU feature Nets must be assigend!")

        imu_out_shape = self.imu_feat_net.get_ouput_shape()
        lidar_out_shape = self.lidar_feat_net.get_output_shape()

        if self.fusion_type == 'cat':
            in_features = lidar_out_shape + imu_out_shape
        else:
            in_features = 512

        self.odom_feat_net = self.create_odom_net(in_features)
        in_features = self.odom_feat_net.get_output_shape()

        if self.p > 0:
            self.drop = nn.Dropout(self.p)
        self.fc_pos = nn.Linear(in_features, 3)
        self.fc_ori = nn.Linear(in_features, 4)

    def create_odom_net(self, in_features):
        odom_type = self.cfg.get('odom-feat-net', 'fc')

        if odom_type == "fc":
            return OdomNetFC(in_features, self.cfg['fc'])
        elif odom_type == "rnn":
            return OdomNetFC(in_features, self.cfg['rnn'])
        else:
            raise ValueError("Wrong odometry feature network specified!")

    def forward(self, x):
        imgs = x[0]
        imus = x[1]

        lidar_feats = self.lidar_feat_net(imgs)
        imu_feats = self.imu_feat_net(imus)

        if self.fusion_type == 'cat':
            fusion_feats = torch.cat((lidar_feats, imu_feats), dim=1)
        else:
            raise NotImplemented("")

        odom_feats = self.odom_feat_net(fusion_feats)

        if self.p > 0.:
            odom_feats = self.drop(odom_feats)

        x_pos = self.fc_pos(odom_feats)
        x_ori = self.fc_ori(odom_feats)
        return x_pos, x_ori


class OdomNetFC(BaseNet):
    def __init__(self, in_features, cfg):
        super(OdomNetFC, self).__init__()

        self.out_features = cfg['size']

        num_layers = len(self.out_features)
        layers = []
        layers.append(nn.Linear(in_features, self.out_features[0]))
        for i in range(1, num_layers):
            l = nn.Linear(self.out_features[i-1], self.out_features[i])
            layers.append(l)

        self.net = nn.ModuleList(layers)

    def get_output_shape(self):
        return self.out_features[-1]

    def forward(self, x):
        y = self.net(x)
        return y


class OdomNetRNN(BaseNet):
    def __init__(self, in_features, cfg):
        super(OdomNetRNN, self).__init__()

        self.rnn_type = cfg.get('type', 'lstm').lower()
        self.hidden_size = cfg['hidden-size']
        self.num_layers = cfg['num-layers']
        self.bidirectional = cfg['bidirectional']

        if self.rnn_type == 'gru':
            self.net = nn.GRU(input_size=in_features, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=self.bidirectional)
        else:
            self.net = nn.LSTM(input_size=in_features, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=self.bidirectional)

    def get_output_shape(self):
        return self.hidden_size

    def forward(self, x):
        y = self.net(x)
        return y
