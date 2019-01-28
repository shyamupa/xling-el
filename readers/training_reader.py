from __future__ import absolute_import
from __future__ import division
import logging
import random
import sys

import numpy as np

import utils.constants as K
from readers.mention_reader import MentionReader, pad_batch


class DataReader(MentionReader):
    def __init__(self, batch_size, args, canddict, istest, iters, loader, dropout, coh_dropout, fpath, num_cands,
                 shuffle=True):
        
        super(DataReader, self).__init__()
        self.num_cands = num_cands
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.word_drop = dropout
        self.coh_drop = coh_dropout
        self.word2idx, self.idx2word = loader.word2idx, loader.idx2word
        self.wid2idx, self.idx2wid = loader.wid2idx, loader.idx2wid
        if args["usedesc"]:
            self.wid2desc = loader.wid2desc
        if args["usetype"]:
            self.num_types = args["ntypes"]
        if args["usecoh"]:
            self.coh2idx, self.idx2coh = loader.coh2idx, loader.idx2coh
            self.num_coh = args["num_coh"]
        self.cands_dict = canddict
        self.num_words = len(self.idx2word)
        self.num_known_entities = args["num_entities"]
        self.word_embedding = loader.embeddings
        self.usecoh = args["usecoh"]
        self.usedesc = args["usedesc"]
        self.usetype = args["usetype"]
        self.usedocbow = args["usedocbow"]
        self.fpath = fpath
        self.prep_loader(fpath=fpath, istest=istest, iters=iters)
        self.istest = istest
        self.cand_counts = 0  # calls made for cand gen
        self.nocands_count = 0  # calls made for cand gen

    # *******************      END __init__      *********************************

    def _next_batch(self):
        """ Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
        start and end are inclusive
        """
        if self.finished:
            raise StopIteration
        # Sentence     = s1 ... m1 ... mN, ... sN.
        # Left Batch   = s1 ... m1 ... mN
        # Right Batch  = sN ... mN ... m1
        left_batch, right_batch = [], []
        doc_bow_batch = []
        types_batch = []

        # Wiki Description nb x 100
        gold_wid_desc_batch = []
        coherence_batch = []

        # Candidate WID idxs and their cprobs
        # First element is always true wid
        cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = [], [], []

        while len(left_batch) < self.batch_size:
            batch_el = len(left_batch)
            m = self._read_mention()
            if m is None:
                self.finished = True
                # logging.info("%d ",len(left_batch))
                if len(left_batch) > 0:
                    return (left_batch, right_batch, doc_bow_batch, gold_wid_desc_batch, types_batch,
                            coherence_batch, cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch)
                else:
                    raise StopIteration
            assert m.idx_version is True

            if m.wid not in self.wid2idx:
                logging.fatal("OOKB entity %s! this should not be here!", m.wid)
                sys.exit(0)

            rv = self.get_candidate_batch(m)
            if rv is None:
                continue

            wids, wid_cprobs, nocands_mask = rv

            # Left and Right context
            left_idxs, right_idxs = self.get_mention_contexts(m)
            if len(left_idxs) == 0 or len(right_idxs) == 0:
                logging.info("empty left or right cxt")
                continue
            left_idxs, right_idxs = self.apply_dropout(left_idxs, self.word_drop), \
                                    self.apply_dropout(right_idxs,self.word_drop)
            left_batch.append(left_idxs)
            right_batch.append(right_idxs)

            if self.usedesc:
                desc = self.make_desc_batch(m=m)
                # gold_wid = wids[0]
                # desc = self.make_descvec_batch(wid=gold_wid)
                gold_wid_desc_batch.append(desc)

            if self.usetype:
                types = self.make_type_batch(m=m)
                types_batch.append(types)

            if self.usecoh:
                coh_idxs = self.make_coherence_batch(m=m)
                coh_idxs = self.apply_dropout(coh_idxs, self.coh_drop)
                coherence_batch.append(coh_idxs)

            if self.usedocbow:
                doc_bow_idxs = self.get_doc_bow(m)
                doc_bow_idxs = self.apply_dropout(doc_bow_idxs, self.word_drop)
                doc_bow_batch.append(doc_bow_idxs)

            cand_wid_idxs_batch.append(wids)
            cand_wid_cprobs_batch.append(wid_cprobs)
            nocands_mask_batch.append(nocands_mask)

        return (left_batch, right_batch, doc_bow_batch, gold_wid_desc_batch, types_batch,
                coherence_batch, cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch)

    def make_coherence_batch(self, m):
        idxs = []
        for coh_str in m.coherence:
            idxs.append(coh_str)
            # if coh_str in self.coh2idx:
            # idxs.append(self.coh2idx[coh_str])
        if not idxs:
            # idxs.append(self.coh2idx[K.OOV_TOKEN])
            idxs.append(K.OOV_ID)
        return idxs

    def make_type_batch(self, m):
        types = [0.0] * self.num_types
        for type_idx in m.types:
            types[type_idx] = 1.0
        return types

    def get_candidate_batch(self, m):
        mention_key = (m.surface, m.wid)
        self.cand_counts += 1
        if mention_key in self.cands_dict:
            wids, wid_cprobs, isgolds = self.cands_dict[mention_key]
            nocands_mask = [0.0] * len(wid_cprobs)
            # Mapped to idxs in wid2idx already
            wids = list(map(lambda wid: self.wid2idx[wid], wids))

            if len(wids) < self.num_cands:
                if not self.istest:
                    wids += self._random_known_ents(self.num_known_entities, self.num_cands - len(wids))
                else:
                    wids += [K.NULL_TITLE_ID] * (self.num_cands - len(wids))
                wid_cprobs += [0.0] * (self.num_cands - len(wid_cprobs))
                isgolds += [0.0] * (self.num_cands - len(isgolds))
                nocands_mask += [-np.inf] * (self.num_cands - len(nocands_mask))
            elif len(wids) > self.num_cands:
                logging.info("WTF!! len(wids) %d dying ...", len(wids))
                wids = wids[:self.num_cands]
                wid_cprobs = wid_cprobs[:self.num_cands]
                isgolds = isgolds[:self.num_cands]
                nocands_mask = isgolds[:self.num_cands]
                # sys.exit(0)

            if isgolds[0] != 1:
                logging.info("first cand is not gold! Exiting ...")
                sys.exit(0)
            return wids, wid_cprobs, nocands_mask
        elif not self.istest:
            # logging.info("fpath is %s", self.fpath)
            # logging.info("train key %s not found! dying ...",mention_key)
            # sys.exit(0)
            return None
            # self.nocands_count += 1
            # if self.nocands_count > 100 and self.nocands_count == self.cand_counts:
            #     logging.info("too many null cands! dying ...")
            #     sys.exit(0)
            # if self.nocands_count % 100000 == 0:
            #     logging.info("seen nocands/total %d/%d = %.2f", self.nocands_count, self.cand_counts,
            #                  self.nocands_count / self.cand_counts)
            # return None
        elif self.istest:
            logging.info("test key %s not found!", mention_key)
            wids = [K.NULL_TITLE_ID] * self.num_cands
            wid_cprobs = [0.0] * self.num_cands
            nocands_mask = [-np.inf] * self.num_cands
            return wids, wid_cprobs, nocands_mask

    def _random_known_ents(self, known_ent_idx, num):
        """
        Given an entity, sample a number of random neg entities from known entity set
        knwn_ent_idx : idx of known entity for which negs are to be sampled
        num : number of negative samples needed
        """
        neg_ents = []
        while len(neg_ents) < num:
            neg = random.randint(0, self.num_known_entities - 1)
            if neg != known_ent_idx:
                neg_ents.append(neg)
        return neg_ents

    def _next_padded_batch(self):
        l_batch, r_batch, doc_bow_batch, \
        gold_wid_desc_batch, \
        types_batch, \
        coherence_batch, \
        wid_batch, \
        wid_cprobs_batch, \
        nocands_mask_batch = self._next_batch()
        l_batch, l_lengths = pad_batch(l_batch, pad_unit=K.PADDING_ID)
        r_batch, r_lengths = pad_batch(r_batch, pad_unit=K.PADDING_ID)
        l_batch = self.embed_batch(l_batch)
        r_batch = self.embed_batch(r_batch)
        l_lengths = np.asarray(l_lengths, dtype=np.long)
        r_lengths = np.asarray(r_lengths, dtype=np.long)

        if self.usedesc:
            gold_wid_desc_batch, _ = pad_batch(gold_wid_desc_batch, pad_unit=K.PADDING_ID)
            gold_wid_descvec_batch = self.embed_batch(gold_wid_desc_batch)
        else:
            gold_wid_descvec_batch = []

        if self.usecoh:
            batch_rcs = []
            batch_vals = []
            for cid, rids in enumerate(coherence_batch):
                tmp = [[cid, rid] for rid in rids]
                batch_rcs += tmp
                batch_vals += [1.0 for _ in rids]
            batch_vals = np.asarray(batch_vals, dtype=np.float64)
            batch_rcs = np.asarray(batch_rcs, dtype=np.long)
            shape = [len(coherence_batch), self.num_coh]
            coherence_batch = batch_rcs, batch_vals, shape
        if self.usedocbow:
            batch_rcs = []
            batch_vals = []
            for cid, (rids,vals) in enumerate(doc_bow_batch):
                tmp = [[cid, rid] for rid in rids]  # col is index in batch, rid is word id
                batch_rcs += tmp
                batch_vals += [v for v in vals]
            batch_vals = np.asarray(batch_vals, dtype=np.float64)
            batch_rcs = np.asarray(batch_rcs, dtype=np.long)
            shape = [len(doc_bow_batch), self.num_docbow]
            doc_bow_batch = batch_rcs, batch_vals, shape

        wid_batch = np.asarray(wid_batch, dtype=np.long)
        wid_cprobs_batch = np.asarray(wid_cprobs_batch, dtype=np.float64)
        nocands_mask_batch = np.asarray(nocands_mask_batch, dtype=np.float64)

        return l_batch, l_lengths, \
               r_batch, r_lengths, \
               doc_bow_batch, \
               gold_wid_descvec_batch, \
               types_batch, \
               coherence_batch, \
               wid_batch, wid_cprobs_batch, nocands_mask_batch

    def embed_batch(self, batch):
        batch = np.array(batch, dtype=np.int32)  # this is needed
        output_batch = self.word_embedding[batch]
        return output_batch

    @staticmethod
    def get_mention_contexts(mention):
        start, end = mention.start_token, mention.end_token
        # Context inclusive of mention surface
        left_idxs = mention.sent_tokens[0:end + 1]
        right_idxs = mention.sent_tokens[start:][::-1]
        return left_idxs, right_idxs

    @staticmethod
    def get_doc_bow(mention):
        # token_length = mention.end_token - mention.start_token
        # left_idxs = left_idxs[:token_length]
        # doc_bow = left_idxs + right_idxs
        doc_bow = [(w,c) for w, c in mention.doc_bow]
        return doc_bow

    def prep_loader(self, iters, istest, fpath):
        self.load_loader(fpath=fpath, istest=istest, iters=iters, shuffle=self.shuffle)

    def make_desc_batch(self, m):
        wid = m.wid
        if wid in self.wid2desc:
            desc = self.wid2desc[wid]
        else:
            desc = self.wid2desc[K.NULL_TITLE_WID]
        return desc

    # def make_descvec_batch(self, wid):
    #     desc = self.wid2descvec[wid]
    #     return desc

    def apply_dropout(self, tokens, dropout):
        if dropout > 0.0:
            for i in range(0, len(tokens)):
                r = random.random()
                if r < self.word_drop:
                    tokens[i] = K.OOV_ID
        return tokens
