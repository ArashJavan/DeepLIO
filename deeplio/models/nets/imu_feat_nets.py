import torch
from torch import nn
from torch.nn import functional as F

from .base_net import BaseNet
from ..misc import get_config_container


class BaseImuFeatNet(BaseNet):
    def __init__(self, cfg):
        super(BaseImuFeatNet, self).__init__()
        self.p = cfg['dropout']
        self.input_size = cfg['input-size']
        self.num_layers = cfg.get('num-layers', 2)
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations
        self.output_shape = None


class ImuFeatFC(BaseImuFeatNet):
    def __init__(self, cfg):
        super(ImuFeatFC, self).__init__(cfg)
        self.hidden_size = cfg.get('hidden-size', [6, 6])
        self.num_layers = len(self.hidden_size)

        layers = [nn.Linear(self.input_size, self.hidden_size[0])]
        for i in range(1, self.num_layers):
            l = nn.Linear(self.hidden_size[i-1], self.hidden_size[i])
            layers.append(l)

        if self.p > 0.:
            layers.append(nn.Dropout(self.p))
        self.net = nn.ModuleList(layers)

        self.output_shape = [1, self.seq_size, self.hidden_size[-1]]

    def forward(self, x):
        batch_size = len(x)
        n_seq = len(x[0]) # all seq. are the same length

        outputs = []
        for b in range(batch_size):
            for s in range(n_seq):
                y = x[b][s]
                for m in self.net:
                    y = F.leaky_relu(m(y), negative_slope=0.01, inplace=True)
                outputs.append(torch.sum(y, dim=0))
        outputs = torch.stack(outputs)
        outputs = outputs.view(batch_size, self.seq_size, -1)
        return outputs


class ImufeatRNN0(BaseImuFeatNet):
    def __init__(self, cfg):
        super(ImufeatRNN0, self).__init__(cfg)
        rnn_type = cfg['type'].lower()
        self.hidden_size = cfg.get('hidden-size', 6)
        self.bidirectional = cfg.get('bidirectional', False)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.p)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.p)

        self.num_dir = 2 if self.bidirectional else 1
        self.output_shape = self.output_shape = [1, self.seq_size, self.hidden_size]

    def forward(self, x):
        batch_size = len(x)
        x_all = [xx for x_ in x for xx in x_]
        lengths = [x_.size(0) for x_ in x_all]
        x_padded = nn.utils.rnn.pad_sequence(x_all)
        s, b, n = x_padded.shape  # seq
        x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, lengths=lengths, enforce_sorted=False)
        out, hidden = self.rnn(x_padded)
        out, lens_unpacked = nn.utils.rnn.pad_packed_sequence(out)
        out = out.view(s, b, self.num_dir, self.hidden_size)
        out = out[-1, :, 0] # many-to-one
        out = out.view(batch_size, self.seq_size, -1)
        return out


class ImuFeatRnn1(BaseImuFeatNet):
    def __init__(self, cfg):
        super(ImuFeatRnn1, self).__init__(cfg)
        rnn_type = cfg['type'].lower()
        self.hidden_size = cfg.get('hidden-size', 6)
        self.bidirectional = cfg.get('bidirectional', False)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.p)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=self.num_layers, bidirectional=self.bidirectional, dropout=self.p)

        self.num_dir = 2 if self.bidirectional else 1
        self.output_shape = self.output_shape = [1, self.seq_size, self.hidden_size]

    def forward(self, x):
        batch_size = len(x)
        x_all = [xx for x_ in x for xx in x_]
        outputs = []
        for xx in x_all:
            s, n = xx.shape
            out, hiden = self.rnn(xx.unsqueeze(1))
            out = out.view(s, 1, self.num_dir, self.hidden_size)
            out = out[-1, :, 0]
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        outputs = outputs.view(batch_size, self.seq_size, -1)
        return outputs

