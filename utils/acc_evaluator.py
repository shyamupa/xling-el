from __future__ import division
import logging
import sys
import torch.nn.functional as F
from utils.el_runner import ELRunner

__author__ = 'Shyam'
import numpy as np


class Overall:
    def __init__(self):
        self.best_scores = ()
        self.best_batch = -1
        self.best_was_updated = False

    def update(self, batch_idx, scores):
        self.best_was_updated = False
        if self.best_scores < scores:
            self.best_scores = scores
            self.best_batch = batch_idx
            self.best_was_updated = True


def write_prediction(outfile, test_iterator, joint_probs):
    old_batch_size = test_iterator.batch_size
    test_iterator.batch_size = 1
    test_examples = []
    while True:
        try:
            batch = test_iterator._next_batch()
            l_batch, r_batch = batch[:2]
            cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = batch[-3:]
            # gold_wids = cand_wid_idxs_batch[:, 0]
            idx2word, idx2wid = test_iterator.idx2word, test_iterator.idx2wid
            for l, r, cand_wids in zip(l_batch, r_batch, cand_wid_idxs_batch):
                l_str = list(map(lambda x: idx2word[x], l))
                r_str = list(map(lambda x: idx2word[x], r))
                wids = list(map(lambda x: idx2wid[x], cand_wids))
                test_examples.append([wids, l_str, r_str])
        except StopIteration:
            break
    test_iterator.batch_size = old_batch_size
    # print(len(test_examples), len(joint_probs))
    out = open(outfile, "w")
    for ex, probs in zip(test_examples, joint_probs):
        # print(ex, probs)
        wids, l_str, r_str = ex
        assert len(wids) == len(probs)
        wid_n_scores = [str(w) + ":" + str(p) for w, p in zip(wids, probs)]
        out.write(" ".join(l_str) + " " + " ".join(r_str) + "\t" + " ".join(wid_n_scores) + "\n")
    out.close()


def compute_acc(y_true, prior_probs, joint_probs):
    assert len(y_true) == len(joint_probs)
    joint_correct = 0

    gold_not_cands = 0
    for gold_label, prior_prob, joint_prob in zip(y_true, prior_probs, joint_probs):
        if prior_prob[0] == 0.0:
            gold_not_cands += 1
            continue
        if np.argmax(joint_prob) == gold_label:
            joint_correct += 1
    # logging.info("acc over %d gold_not_cands %d", len(prior_probs), gold_not_cands)
    # cxt_score = correct / len(prior_probs)
    joint_score = joint_correct / len(prior_probs)
    ceiling = (len(prior_probs) - gold_not_cands) / len(prior_probs)
    logging.info("joint %.4f ceil %.4f",
                 joint_score,
                 ceiling)
    return joint_score


def compute_joint_probs(cxt_probs, cand_probs, interpol=1.0):
    joint_probs = []
    for (cand_prob_dist, cxt_prob_dist) in zip(cand_probs, cxt_probs):
        if sum(cand_prob_dist) == 0.0:
            joint_probs.append(cxt_prob_dist)
            continue
        cand_prob_dist = [c / sum(cand_prob_dist) for c in cand_prob_dist]
        cxt_prob_dist = [c / sum(cxt_prob_dist) for c in cxt_prob_dist]
        if interpol == 1.0:
            joint_prob = [(x + y - x * y) for (x, y) in zip(cand_prob_dist, cxt_prob_dist)]
        else:
            interpol_cand_prob_dist = [x ** interpol for x in cand_prob_dist]
            joint_prob = [(x + y - x * y) for (x, y) in
                          zip(interpol_cand_prob_dist, cxt_prob_dist)]
        Z = sum(joint_prob)
        if Z != 0.0:
            joint_prob = [float(x) / Z for x in joint_prob]
        joint_probs.append(joint_prob)
    return joint_probs


class AccEvaluator:
    def __init__(self):
        self.best = 0
        self.best_batch = 0
        self.best_cxt = 0
        self.best_cxt_batch = 0
        self.best_was_updated = False

    def get_best(self):
        return self.best

    def get_best_batch(self):
        return self.best_batch

    def get_cxt_best(self):
        return self.best_cxt

    def get_cxt_best_batch(self):
        return self.best_cxt_batch

    def best_was_updated(self):
        return self.best_was_updated

    def test(self, args, model, test_iterator):
        model.eval()
        all_model_probs = []
        cand_wid_cprobs = []
        for bid, batch in enumerate(test_iterator):
            cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = batch[-3:]
            if len(cand_wid_cprobs_batch) < test_iterator.batch_size:
                model.batch_size = len(cand_wid_cprobs_batch)
            # nb x ncands
            batch = model.prepare_batch(batch, istest=True)
            logits = model.forward(batch)
            cxt_logits = logits["cxt_logits"]
            cxt_probs = F.softmax(cxt_logits)
            cand_wid_cprobs.extend(cand_wid_cprobs_batch.tolist())
            if model.args["cuda"]:
                model_probs = cxt_probs.data.cpu().numpy()
            else:
                model_probs = cxt_probs.data.numpy()
            all_model_probs += model_probs.tolist()

        all_golds = len(all_model_probs) * [0]
        joint_probs = compute_joint_probs(cxt_probs=all_model_probs, cand_probs=cand_wid_cprobs)
        model.batch_size = args["batch_size"]
        model.train()
        test_iterator.reset()
        if args["dump"]:
            write_prediction(args["dump"], test_iterator, joint_probs)
            test_iterator.reset()
        else:
            compute_acc(y_true=all_golds,
                        prior_probs=cand_wid_cprobs,
                        joint_probs=joint_probs)

    def do_eval(self, args, runner, model, test_iterator, batch_idx):
        self.best_was_updated = False
        model.eval()
        all_model_probs = []
        all_mask = []
        all_hard_mask = []
        cand_wid_cprobs = []
        eval_loss = []
        for bid, batch in enumerate(test_iterator):
            cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = batch[-3:]
            if len(cand_wid_cprobs_batch) < test_iterator.batch_size:
                model.batch_size = len(cand_wid_cprobs_batch)
            # nb x ncands
            batch = model.prepare_batch(batch, istest=True)
            logits = model.forward(batch)
            cxt_logits = logits["cxt_logits"]
            loss = runner.get_total_loss(logits=logits, batch=batch)
            eval_loss.append(loss.data[0])
            # nb
            # nz = np.count_nonzero(cand_wid_cprobs_batch, axis=1) > 1
            # mask = nz.astype(np.int)
            # all_mask += mask.tolist()

            # prior_preds = cand_wid_cprobs_batch.argmax(axis=1)
            # all_prior_preds += prior_preds.tolist()

            # hard_mask = (prior_preds > 0).astype(np.int)
            # all_hard_mask += hard_mask.tolist()
            cxt_probs = F.softmax(cxt_logits)
            cand_wid_cprobs.extend(cand_wid_cprobs_batch.tolist())
            if model.args["cuda"]:
                model_probs = cxt_probs.data.cpu().numpy()
            else:
                model_probs = cxt_probs.data.numpy()
            all_model_probs += model_probs.tolist()

        all_golds = len(all_model_probs) * [0]
        # for interpol in [1.0]:
        # logging.info("interpol %.2f",interpol)
        joint_probs = compute_joint_probs(cxt_probs=all_model_probs,
                                          cand_probs=cand_wid_cprobs,
                                          interpol=1.0)
        model_score = compute_acc(y_true=all_golds,
                                  prior_probs=cand_wid_cprobs,
                                  joint_probs=joint_probs)
        logging.info("eval loss:%.3f", np.mean(eval_loss))
        model.batch_size = args["batch_size"]
        model.train()
        test_iterator.reset()
        if args["dump"]:
            write_prediction(args["dump"], test_iterator, joint_probs)
            test_iterator.reset()
        if model_score > self.best:
            self.best = model_score
            self.best_batch = batch_idx
            self.best_was_updated = True
            if args["save"]:
                ELRunner.save_checkpoint({
                    'state_dict': model.state_dict(),
                    'cxt_score': self.best_cxt,
                    'optimizer': runner.optimizer.state_dict(),
                }, filename=args["save"], is_best=True)
        if cxt_score > self.best_cxt:
            self.best_cxt = cxt_score
            self.best_cxt_batch = batch_idx
        return model_score, cxt_score
