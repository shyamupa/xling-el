import torch.nn as nn

# import cuda_functional as MF
from model.conv_sem_sim import ConvSemSim


class DescEncoder(nn.Module):
    def __init__(self, args):
        super(DescEncoder, self).__init__()
        self.encoder = ConvSemSim(inp_type="desc",
                                  embed_dim=args["wdim"],
                                  filter_num=args["filter_num"],
                                  filter_sizes=args["filter_sizes"],
                                  pooling="avg")
        self.linear_weights_desc = nn.Linear(in_features=args["filter_num"] * len(args["filter_sizes"]),
                                             out_features=args["ncands"])
        nn.init.xavier_normal(self.linear_weights_desc.weight)
        self.fflayer_desc = nn.Linear(in_features=args["filter_num"] * len(args["filter_sizes"]),
                                      out_features=args["hdim"])
        nn.init.xavier_normal(self.fflayer_desc.weight)

    def forward(self, desc_vecs):
        # if self.use_rnn:
        #     pass
        # elif self.use_conv:
        #     desc_repr = self.encoder.forward(desc_vecs)
        # else:  # bag of averaged words
        #     desc_repr = torch.mean(desc_vecs)

        # desc_vecs = nb x maxlen x wdim
        desc_repr = self.encoder.forward(desc_vecs)
        # nb x dim
        out = self.fflayer_desc.forward(desc_repr)
        return out
