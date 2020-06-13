import os
import torch
from torch import nn
from torch.nn import functional as F

from deeplio.common.logger import get_app_logger
from .base_net import BaseNet, BaseDeepLIO


class DeepIO(BaseDeepLIO):
    def __init__(self, cfg):
        super(DeepIO, self).__init__()

        self.logger = get_app_logger()

        self.cfg = cfg['deepio']
        self.p = self.cfg.get('dropout', 0.)
        self.feat_net = None

    def initialize(self):
        if self.feat_net is None:
            raise ValueError("{}: feature net is not defined!".format(self.name))

        in_features = self.feat_net.get_output_shape()

        if self.p > 0:
            self.drop = nn.Dropout(self.p)
        self.fc_pos = nn.Linear(in_features, 3)
        self.fc_ori = nn.Linear(in_features, 4)

    def forward(self, x):
        x = self.feat_net(x)

        #x = F.relu(self.fc1(x), inplace=True)
        #x = self.bn1(x)
        if self.p > 0.:
            x = self.drop(x)

        x_pos = self.fc_pos(x)
        x_ori = self.fc_ori(x)
        return x_pos, x_ori

    @property
    def name(self):
        res = "{}_{}".format(self.__class__.__name__, self.feat_net.__class__.__name__)
        return res

    def get_feat_networks(self):
        return [self.feat_net]


class DeepIOFeat0(BaseNet):
    def __init__(self, cfg):
        super(DeepIOFeat0, self).__init__()

        self.input_size = cfg['input-size']
        self.hidden_size = cfg.get('hidden-size', [6, 6])
        num_layers = cfg.get('num-layers', 2)

        layers = [nn.Linear(self.input_size, self.hidden_size[0])]
        for i in range(1, num_layers):
            l = nn.Linear(self.hidden_size[i-1], self.hidden_size[i])
            layers.append(l)
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        n_batches = len(x)
        n_seq = len(x[0]) # all seq. are the same length

        outputs = []
        for b in range(n_batches):
            for s in range(n_seq):
                y = x[b][s]
                for m in self.net:
                    y = F.relu(m(y))
                outputs.append(torch.sum(y, dim=0))
        outputs = torch.stack(outputs)
        return outputs

    def get_output_shape(self):
        return self.hidden_size[-1]


class DeepIOFeat1(BaseNet):
    def __init__(self, cfg):
        super(DeepIOFeat1, self).__init__()
        rnn_type = cfg['type'].lower()
        num_layers = cfg.get('num-layers', 2)
        self.input_size = cfg['input-size']
        self.hidden_size = cfg.get('hidden-size', 6)
        self.bidirectional = cfg.get('bidirectional', False)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=num_layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=num_layers, bidirectional=self.bidirectional)

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        x_all = [xx for x_ in x for xx in x_]
        lengths = [x_.size(0) for x_ in x_all]
        x_padded = nn.utils.rnn.pad_sequence(x_all)
        s, b, n = x_padded.shape  # seq
        x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, lengths=lengths, enforce_sorted=False)
        out, hidden = self.rnn(x_padded)
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out)
        out = out.view(s, b, self.num_dir, self.hidden_size)
        out = out[-1, :, 0]
        return out

    def get_output_shape(self):
        return self.hidden_size


class DeepIOFeat11(nn.Module):
    def __init__(self, cfg):
        super(DeepIOFeat11, self).__init__()
        rnn_type = cfg['type'].lower()
        num_layers = cfg.get('num-layers', 2)
        self.input_size = cfg['input-size']
        self.hidden_size = cfg.get('hidden-size', [6, 6])
        self.bidirectional = cfg.get('bidirectional', False)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size[0],
                              num_layers=num_layers, bidirectional=self.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0],
                               num_layers=num_layers, bidirectional=self.bidirectional)

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        x_all = [xx for x_ in x for xx in x_]
        outputs = []
        for xx in x_all:
            s, n = xx.shape
            out, hiden = self.rnn(xx.unsqueeze(1))
            out = out.view(s, 1, self.num_dir, self.hidden_size[0])
            out = out[-1, :, 0]
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        return outputs

    def get_output_shape(self):
        return self.hidden_size