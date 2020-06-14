from torch import nn
from torch.nn import functional as F

from .base_net import BaseNet
from ..misc import get_config_container


class OdomFeatFC(BaseNet):
    def __init__(self, in_features, cfg):
        super(OdomFeatFC, self).__init__()
        self.input_size = in_features
        self.hidden_size = cfg.get('hidden-size', [256, 128])
        self.p = cfg.get('dropout', 0.)
        num_layers = len(self.hidden_size)
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations

        layers = [nn.Linear(self.input_size, self.hidden_size[0])]
        for i in range(1, num_layers):
            layers.append(nn.Linear(self.hidden_size[i - 1], self.hidden_size[i]))
        if self.p > 0.:
            self.layers.append(nn.Dropout(self.p))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """

        :param x: input of dim [BxTxN]
        :return:
        """
        b, s, n = x.shape
        x = x.view(b*s, n)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True)
        x = x.view(b, s, -1)
        return x

    def get_output_shape(self):
        # all layer should have [BxTx..] as input and output
        return [1, 1, self.hidden_size[-1]]


class OdomFeatRNN(BaseNet):
    def __init__(self, in_features, cfg):
        super(OdomFeatRNN, self).__init__()
        rnn_type = cfg['type'].lower()
        num_layers = cfg.get('num-layers', 2)
        self.hidden_size = cfg.get('hidden-size', 6)
        self.p = cfg.get('dropout', 0.)
        self.bidirectional = cfg.get('bidirectional', False)
        self.input_size = in_features
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.combinations = self.cfg_container.combinations

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=num_layers, bidirectional=self.bidirectional,
                              batch_first=True, dropout=self.p)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True, dropout=self.p)

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        """

        :param x: input, dim= [BxTxN]
        :return:
        """
        # reorder to seq. first. Since it seems to have better comput. performance, then batch first
        b, s, n = x.shape
        out, _ = self.rnn(x)
        out = out.view(b, s, self.num_dir, self.hidden_size)
        out = out[:, :, 0].contiguous()
        return out

    def get_output_shape(self):
        return [1, 1, self.hidden_size]
