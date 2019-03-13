__author__ = 'Shyam'
import torch
from torch.autograd import Variable as V
import torch.nn as nn
# import cuda_functional as MF
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq
from model.conv_sem_sim import ConvSemSim
import logging
from torch.nn.parameter import Parameter


class ContextEncoder(nn.Module):
    def __init__(self, args):
        super(ContextEncoder, self).__init__()
        self.use_lstm = args["uselstm"]
        self.rnn_type = args["cell"]
        self.embed_dim = args["wdim"]
        self.device_id = args["device_id"]
        self.hidden_dim = args["hdim"]
        self.args = args
        if args["usecoh"]:
            self.coh_matrix = Parameter(torch.FloatTensor(args["num_coh"], args["hdim"]))
            # self.coh_matrix = nn.Linear(in_features=args["num_coh"],out_features=args["hdim"], bias=False)
            self.fflayer_cxt = nn.Linear(in_features=2 * args["hdim"],
                                         out_features=args["hdim"])
            nn.init.xavier_normal(self.fflayer_cxt.weight)
        if self.use_lstm:
            logging.info("Using rnn for cxt encoder of type %s", args["cell"])
            if self.rnn_type == "lstm":
                self.left_cxt_enc = nn.LSTM(input_size=args["wdim"],
                                            hidden_size=args["hdim"],
                                            num_layers=1, batch_first=True)
                self.right_cxt_enc = nn.LSTM(input_size=args["wdim"],
                                             hidden_size=args["hdim"],
                                             num_layers=1, batch_first=True)
            elif self.rnn_type == "gru":
                self.left_cxt_enc = nn.GRU(input_size=args["wdim"],
                                           hidden_size=args["hdim"],
                                           num_layers=1, batch_first=True)
                self.right_cxt_enc = nn.GRU(input_size=args["wdim"],
                                            hidden_size=args["hdim"],
                                            num_layers=1, batch_first=True)
            elif self.rnn_type == "sru":
                self.left_cxt_enc = MF.SRU(input_size=args["wdim"],
                                           hidden_size=args["hdim"],
                                           num_layers=1, batch_first=True)
                self.right_cxt_enc = MF.SRU(input_size=args["wdim"],
                                            hidden_size=args["hdim"],
                                            num_layers=1, batch_first=True)
            self.fflayer_loc_cxt = nn.Linear(in_features=2 * args["hdim"],
                                             out_features=args["hdim"])
            nn.init.xavier_normal(self.fflayer_loc_cxt.weight)

        else:
            # logging.info("Using conv for cxt encoder")
            self.left_cxt_enc = ConvSemSim(inp_type="left_cxt",
                                           embed_dim=args["wdim"],
                                           filter_num=args["filter_num"],
                                           filter_sizes=args["filter_sizes"],
                                           pooling="avg")

            self.right_cxt_enc = ConvSemSim(inp_type="right_cxt",
                                            embed_dim=args["wdim"],
                                            filter_num=args["filter_num"],
                                            filter_sizes=args["filter_sizes"],
                                            pooling="avg")
            self.linear_weights_cxt = nn.Linear(in_features=args["filter_num"] * len(args["filter_sizes"]),
                                                out_features=args["ncands"])
            nn.init.xavier_normal(self.linear_weights_cxt.weight)
            self.fflayer_loc_cxt = nn.Linear(in_features=2 * args["filter_num"] * len(args["filter_sizes"]),
                                             out_features=args["hdim"])
            nn.init.xavier_normal(self.fflayer_loc_cxt.weight)

    def forward(self, context):
        l_batch, l_lengths, r_batch, r_lengths, coherence_batch = context

        if self.use_lstm:
            # nb x hdim
            l_cxt_emb = self.get_last_rnn(self.rnn_type, self.left_cxt_enc,
                                           l_batch, l_lengths)
            # nb x hdim
            r_cxt_emb = self.get_last_rnn(self.rnn_type, self.right_cxt_enc,
                                           r_batch, r_lengths)
        else:
            # nb x hdim
            l_cxt_emb = self.left_cxt_enc.forward(l_batch)
            # nb x hdim
            r_cxt_emb = self.right_cxt_enc.forward(r_batch)

        # nb x 2*hdim
        cat_local_cxt = torch.cat([l_cxt_emb, r_cxt_emb], 1)
        # nb x 2*hdim --> nb x hdim
        local_cxt = self.fflayer_loc_cxt.forward(cat_local_cxt)

        if not self.args["usecoh"]:
            return local_cxt
        # nb x numcoh --> nb x hdim
        doc_emb = torch.mm(coherence_batch, self.coh_matrix)
        # doc_emb = self.coh_matrix.forward(coherence_batch)

        # nb x 2*hdim
        cat_cxt = torch.cat([local_cxt, doc_emb], 1)
        # nb x hdim
        out = self.fflayer_cxt.forward(cat_cxt)
        return out

    def init_hidden(self):
        if self.rnn_type == 'lstm':
            return V(torch.zeros(1 * 2, self.batch_size, self.hidden_dim)), \
                   V(torch.zeros(1 * 2, self.batch_size, self.hidden_dim))
        else:
            return V(torch.zeros(1 * 2, self.batch_size, self.hidden_dim))

    def zero_hid(self, nlayers, bs, hsize):
        hid = V(torch.zeros(nlayers, bs, hsize))
        hid = self._cuda(hid)
        return hid

    def get_last_rnn(self, rnn_type, rnn, embeds, input_lengths):

        bs = embeds.size()[0]

        if rnn_type == "lstm":
            (h0, c0) = (self.zero_hid(1, bs, rnn.hidden_size),
                        self.zero_hid(1, bs, rnn.hidden_size))
            rnn_init = (h0, c0)
        else:
            h0 = self.zero_hid(1, bs, rnn.hidden_size)
            rnn_init = h0

        s_lengths, so_idxs = torch.sort(input_lengths, dim=0, descending=True)

        # nb x maxlen x wdim (just reorder)
        embeds = embeds[so_idxs.data]

        try:
            # Pack
            packed_x = packseq(embeds, list(s_lengths.data), batch_first=True)
        except ValueError as e:
            print(s_lengths)
            raise e
            sys.exit(0)
        # Forward propagate RNN
        out, rnn_outs = rnn(packed_x, rnn_init)

        if rnn_type=='lstm':
            (h, c) = rnn_outs
        else:
            h = rnn_outs

        h = h.squeeze(0)
        h_unsort = self._cuda(V(h.data.new(*h.data.size())))

        soridx2d = so_idxs.unsqueeze(1).expand(h_unsort.size())
        lstm_output_h = h_unsort.scatter_(0, soridx2d, h)

        return lstm_output_h

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m
