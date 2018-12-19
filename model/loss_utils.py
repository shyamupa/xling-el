__author__ = 'Shyam'

import torch
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F

def compute_logsumexp_loss(feats_type, logits):
    """
    use this for both description and context loss
    :param feats_type: just for book-keeping
    :param logits: nb x ncands (scores for each cand)
    :return:
    """
    # nb
    batch_size = logits.size(0)
    gold = V(torch.zeros(batch_size).long())
    loss = nn.CrossEntropyLoss()
    # scores = nb x ncands
    loss = loss(input=logits, target=gold)
    return loss


def compute_ranking_loss(logits, ncands, margin=5.0):
    logsm = nn.LogSoftmax()
    scores = logsm(logits)
    batch_size = logits.size(0)
    rank_loss = nn.MarginRankingLoss(margin=margin)
    # nb x ( num_cands - 1 )
    neg_scores = scores[:, 1:]
    # print("neg_scores",neg_scores.size())
    # nb --> nb x 1 --> nb x ( num_cands - 1 )
    gold_scores = scores[:, 0].unsqueeze(1).expand_as(neg_scores) # scores[:, 0].repeat(1, (ncands - 1))
    # print("gold_scores",gold_scores.size())
    # nb x 2*( num_cands - 1 )
    x1 = torch.cat([gold_scores, neg_scores], 1)
    # print("x1", x1.size())
    x2 = torch.cat([neg_scores, gold_scores], 1)
    # print("x2", x2.size())
    y = torch.cat([torch.ones(batch_size, ncands - 1) ,  -1 * torch.ones(batch_size, ncands - 1)],1)
    # print("y", y.size())
    y = V(y)
    loss = rank_loss.forward(x1, x2, y)
    return loss


def compute_type_loss(pred_types, gold_types):
    loss = F.binary_cross_entropy_with_logits(pred_types, gold_types)
    return loss
