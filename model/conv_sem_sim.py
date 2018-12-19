__author__ = 'Shyam'

import torch
import torch.nn as nn
import torch.nn.functional as F

pool_type = {"avg": F.avg_pool1d, "max": F.max_pool1d}


class ConvSemSim(nn.Module):
    def __init__(self, inp_type, embed_dim, filter_num, filter_sizes, pooling):
        super(ConvSemSim, self).__init__()
        self.inp_type = inp_type
        self.pooling = pool_type[pooling]
        D = embed_dim
        Ci = 1
        Co = filter_num  # 150
        Ks = filter_sizes  # [5]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D)) for K in Ks])
        # initialize each conv using xavier
        for conv in self.convs:
            nn.init.xavier_normal(conv.weight)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x:  nb x maxlen x dim, batch of embedded words of maxlen
        :return: nb x hdim representation computed by max pooled convolutions
        """
        # logging.info('x in conv %s  %s', x.size(), type(x.data))
        # x is shape nb x maxlen x dim
        x = x.unsqueeze(1)
        # nb x Ci x maxlen x dim
        x = [F.relu(conv(x)) for conv in self.convs]
        # [ (nb x Co x maxlen-k+1 x 1), ...]
        x = [i.squeeze(3) for i in x]
        # [ (nb x Co x maxlen-k+1), ...]
        x = [self.pooling(i, i.size(2)).squeeze(2) for i in x]
        # [ (nb x Co), ...]
        x = torch.cat(x, 1)
        # nb x num_filter_sizes * Co
        return x


if __name__ == '__main__':
    ConvSemSim(inp_type="left_cxt",
               embed_dim=100,
               filter_num=50,
               filter_sizes=[5],
               pooling="avg")
