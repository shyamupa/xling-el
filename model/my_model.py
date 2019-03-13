from model.context_encoder import ContextEncoder
from model.desc_encoder import DescEncoder
import math

__author__ = 'Shyam'
import torch
from torch.autograd import Variable as V
import torch.nn as nn
# import cuda_functional as MF
from torch import FloatTensor as MyTensor
from torch.sparse import FloatTensor as MySparseTensor
import logging


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.batch_size = args["batch_size"]
        self.embed_dim = args["wdim"]
        self.device_id = args["device_id"]
        self.hidden_dim = args["hdim"]
        self.ncands = args["ncands"]  # 20
        self.num_entities = args["num_entities"]
        # need sparse updates for fast training, tho this limits to certain kinds of optimizer only (maybe fixed in
        # next release)
        if args["usetype"]:
            self.ntypes = args["ntypes"]
            self.type_embeddings = nn.Embedding(num_embeddings=args["ntypes"],
                                                sparse=True if args["optimizer"] == "sgd" else False,
                                                embedding_dim=args["hdim"])
            nn.init.xavier_normal(self.type_embeddings.weight)

        self.entity_emb = nn.Embedding(num_embeddings=args["num_entities"],
                                       sparse=True if args["optimizer"] == "sgd" else False,
                                       embedding_dim=args["hdim"])
        nn.init.xavier_normal(self.entity_emb.weight)

        if args["usedocbow"]:
            self.fflayer_dbow = nn.Linear(in_features=args["num_words"],
                                          out_features=self.hidden_dim)
            nn.init.xavier_normal(self.fflayer_dbow.weight)
        self.log_sigma_sq = args["logsigsq"]  # b/w -2 and 5
        init_val = math.sqrt(math.e ** self.log_sigma_sq)
        # logging.info("init sigma is %f", init_val)
        # self.ecloss_sigma = torch.nn.Parameter(MyTensor([init_val]))
        # self.etloss_sigma = torch.nn.Parameter(MyTensor([init_val]))
        # self.ctloss_sigma = torch.nn.Parameter(MyTensor([init_val]))

        self.cxt_encoder = ContextEncoder(args=args)

        if args["usedesc"]:
            self.desc_encoder = DescEncoder(args=args)

        if args["usecoh"]:
            self.fflayer_coh = nn.Linear(in_features=args["num_coh"],
                                         out_features=self.hidden_dim)
            nn.init.xavier_normal(self.fflayer_coh.weight)

    def predict_type_from_vec(self, cxt_vec):
        # nb x hdim --> nb x ntypes x hdim
        nb, hdim = cxt_vec.size(0), cxt_vec.size(1)
        cxt_vec = cxt_vec.unsqueeze(1).expand(nb, self.ntypes, self.hidden_dim)

        type_vecs = self.type_embeddings.weight

        # ntypes x hdim --> nb x ntypes x hdim
        tmp2 = type_vecs.unsqueeze(0).expand(nb, self.ntypes, self.hidden_dim)

        # nb x ntypes x hdim --> nb x ntypes
        sigmoid_logits = tmp2.mul(cxt_vec).sum(2)
        return sigmoid_logits

    def prepare_batch(self, batch, istest=False):

        volatile = True if istest else False
        # logging.info("istest is %s and volatile is %s", istest, volatile)

        l_batch, l_lengths, \
        r_batch, r_lengths, \
        doc_bow_batch, \
        truewid_descvec_batch, \
        types_batch, \
        coherence_batch, \
        wids_batch, wid_cprobs_batch, nocands_mask_batch = batch

        # Do not mess with type casting here. torch.from_numpy is SLOW
        l_batch = MyTensor(l_batch)
        l_lengths = torch.LongTensor(l_lengths)
        r_batch = MyTensor(r_batch)
        r_lengths = torch.LongTensor(r_lengths)
        wids_batch = torch.LongTensor(wids_batch)
        # nocands_mask_batch = MyTensor(nocands_mask_batch)

        if self.args["usecoh"]:
            batch_rcs, batch_vals, shape = coherence_batch
            batch_rcs = torch.LongTensor(batch_rcs)
            batch_vals = MyTensor(batch_vals)
            coherence_batch = MySparseTensor(batch_rcs.t(), batch_vals, torch.Size(shape))

        if self.args["usedesc"]:
            truewid_descvec_batch = MyTensor(truewid_descvec_batch)
        if self.args["usetype"]:
            types_batch = MyTensor(types_batch)

        if self.args["cuda"]:
            devid = self.args["device_id"]
            l_batch = l_batch.cuda(device=devid)
            l_lengths = l_lengths.cuda(device=devid)
            r_batch = r_batch.cuda(device=devid)
            r_lengths = r_lengths.cuda(device=devid)
            wids_batch = wids_batch.cuda(device=devid)
            # nocands_mask_batch = nocands_mask_batch.cuda(device=devid)
            if self.args["usecoh"]:
                coherence_batch = coherence_batch.cuda(device=devid)
            if self.args["usedesc"]:
                truewid_descvec_batch = truewid_descvec_batch.cuda(device=devid)
            if self.args["usetype"]:
                types_batch = types_batch.cuda(device=devid)

        l_batch, l_lengths = V(l_batch, volatile=volatile), V(l_lengths, volatile=volatile)
        r_batch, r_lengths = V(r_batch, volatile=volatile), V(r_lengths, volatile=volatile)
        wids_batch = V(wids_batch, volatile=volatile)
        # nocands_mask_batch = V(nocands_mask_batch)
        if self.args["usecoh"]:
            coherence_batch = V(coherence_batch, volatile=volatile)
        if self.args["usedesc"]:
            truewid_descvec_batch = V(truewid_descvec_batch, volatile=volatile)
        if self.args["usetype"]:
            types_batch = V(types_batch, volatile=volatile)

        batch = l_batch, l_lengths, \
                r_batch, r_lengths, \
                truewid_descvec_batch, \
                types_batch, \
                coherence_batch, \
                wids_batch, wid_cprobs_batch, nocands_mask_batch
        return batch

    def forward(self, batch):
        l_batch, l_lengths, \
        r_batch, r_lengths, \
        truewid_descvec_batch, \
        types_batch, \
        coherence_batch, \
        wids_batch, wid_cprobs_batch, nocands_mask_batch = batch

        context = l_batch, l_lengths, r_batch, r_lengths, coherence_batch

        # nb x cands x hdim
        all_entemb = self.entity_emb(wids_batch)
        # nb x 1 x hdim
        true_entemb = all_entemb[:, 0, :]

        # nb x hdim
        cxt_vec = self.cxt_encoder.forward(context=context)
        # nb x hdim --> nb x 1 x hdim --> nb x cands x hdim
        exp_cxt_vec = cxt_vec.unsqueeze(1).expand_as(all_entemb)
        # nb x cands x hdim --> nb x cands
        cxt_logits = exp_cxt_vec.mul(all_entemb).sum(2)

        logits = {"cxt_logits": cxt_logits}

        if self.args["usedesc"]:
            # nb x maxlen x wdim --> nb x hdim
            desc_vec = self.desc_encoder.forward(truewid_descvec_batch)
            # nb x hdim --> nb x 1 x hdim --> nb x cands x hdim
            exp_desc_vec = desc_vec.unsqueeze(1).expand_as(all_entemb)
            # nb x cands x hdim --> nb x cands
            desc_logits = exp_desc_vec.mul(all_entemb).sum(2)
            logits["desc_logits"] = desc_logits

        if self.args["usetype"]:
            type_given_cxt_logits = self.predict_type_from_vec(cxt_vec)
            type_given_ent_logits = self.predict_type_from_vec(true_entemb.squeeze(1))
            logits["type_given_cxt_logits"] = type_given_cxt_logits
            logits["type_given_ent_logits"] = type_given_ent_logits
        return logits

    def infer(self, eval_batch):
        pass
