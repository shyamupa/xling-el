import logging
import random
import sys
import time

import numpy as np
import torch

from model.my_model import MyModel
from readers.training_reader import DataReader
from utils.acc_evaluator import AccEvaluator, Overall
from utils.arguments import PARSER
from utils.el_runner import ELRunner
from utils.misc_vocab_loader import VocabLoader

__author__ = 'Shyam'

optimizers = {"adam": torch.optim.Adam,
              "adagrad": torch.optim.Adagrad,
              "sgd": torch.optim.SGD,
              }


def test_train_reader(reader, model):
    prep_time = 0
    read_time = 0
    start = 0
    for idx, batch in enumerate(reader):
        # l_batch, l_lengths, \
        # r_batch, r_lengths, \
        # gold_wid_desc_vec_batch, \
        # types_batch, \
        # coherence_batch, \
        # wid_batch, wid_cprobs_batch, _ = batch
        end = time.time()
        read_time += end - start
        start = time.time()
        # model.prepare_batch(batch)
        end = time.time()
        prep_time += end - start
        if idx > 0 and idx % 20 == 0:
            logging.info("seen %d batches", idx)
            logging.info("read time %.2f", read_time)
            logging.info("prep time %.2f", prep_time)
            read_time = 0
            prep_time = 0
            # print("l_batch", l_batch.shape)
            # print("r_batch", r_batch.shape)
            # print("types_batch", types_batch.shape)
            # print("wid_batch", wid_batch.shape)
            # print("wid_cprobs_batch", wid_cprobs_batch.shape)
    print(idx, "batches seen")


def test_test_reader(reader):
    all_golds = []
    for bid, batch in enumerate(reader):
        cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = batch[-3:]
        # nb x ncands
        all_golds += len(cand_wid_cprobs_batch) * [0]
        if reader.epochs_done() > 0:
            print("test data size ~", bid * len(batch[0]))
            break
    print(len(all_golds))


def main(args):
    # args["usecoh"] = True
    logging.info(args)
    loader = VocabLoader()
    loader.load_word2idx(word2idx_pkl_path=args["vocabpkl"])
    loader.load_embeddings(embeddings_pkl_path=args["vecpkl"])
    loader.load_wid2idx(kb_file=args["kb_file"])
    args["num_entities"] = len(loader.wid2idx)
    args["num_words"] = len(loader.word2idx)
    logging.info("num_entities %d", args["num_entities"])
    loader.load_test_cand_dict(path=args["ttcands"])
    if args["vacands"]:
        loader.load_val_cand_dict(path=args["vacands"])
    if args["usecoh"]:
        loader.load_coh2idx(path=args["cohstr"])
        args["num_coh"] = len(loader.coh2idx)

    if args["usetype"]:
        loader.load_type_vocab(path=args["type_vocab"])
        args["ntypes"] = len(loader.type2idx)

    args["filter_sizes"] = list(map(int, args["filter_sizes"].split(",")))
    # logging.info("filter_sizes %d", args["filter_sizes"])

    test_data = DataReader(args=args, batch_size=args["batch_size"],
                           iters=1,
                           istest=True, fpath=args["ftest"],
                           dropout=0.0, coh_dropout=0.0,
                           canddict=loader.test_cand_dict, loader=loader,
                           num_cands=args["ncands"], shuffle=False)
    if args["fdev"]:
        dev_data = DataReader(args=args, batch_size=args["batch_size"],
                              iters=1,
                              istest=True, fpath=args["fdev"],
                              dropout=0.0, coh_dropout=0.0,
                              canddict=loader.val_cand_dict, loader=loader,
                              num_cands=args["ncands"], shuffle=False)
    else:
        dev_data = None

    model = MyModel(args=args)

    if args["cuda"]:
        logging.info("Putting things on GPU ...")
        model.cuda(args["device_id"])

    # for name, param in model.named_parameters():
    #     logging.info("%s %s %s", name, type(param.data), param.data.size())

    params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
    optim_type = optimizers[args["optimizer"]]
    optimizer = optim_type(params=params_to_opt, lr=args["lr"])
    test_evaler = AccEvaluator()
    dev_evaler = AccEvaluator()
    dev2_evaler = AccEvaluator()
    dev3_evaler = AccEvaluator()

    if args["restore"]:
        ELRunner.load_checkpoint(model=model, optimizer=optimizer, ckpt_path=args["restore"])
        test_evaler.test(args=args, model=model, test_iterator=test_data)
        if args["fdev"]:
            dev_evaler.test(args=args, model=model, test_iterator=dev_data)
    else:
        loader.load_train_cand_dict(path=args["trcands"])
        train_data = DataReader(args=args, batch_size=args["batch_size"],
                                iters=1,
                                istest=False, fpath=args["ftrain"],
                                dropout=args["wdrop"], coh_dropout=args["cdrop"],
                                canddict=loader.trval_cand_dict, loader=loader,
                                num_cands=args["ncands"])

        model.train()
        logging.info("starting training ...")
        runner = ELRunner(args=args, model=model,
                          optimizer=optimizer,
                          maxsteps=args["maxsteps"],
                          train_it=train_data,
                          test_it=test_data,
                          test_evaler=test_evaler,
                          dev_it=dev_data,
                          dev_evaler=dev_evaler,
                          dev2_it=None,
                          dev2_evaler=dev2_evaler,
                          dev3_it=None,
                          dev3_evaler=dev3_evaler,
                          overall=Overall(),
                          logfreq=args["logfreq"], evalfreq=args["evalfreq"])
        runner.run(maxepoch=args["iters"], train_it=train_data)


if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    if args["cuda"]:
        if not torch.cuda.is_available():
            print("cuda not found. exiting ...")
            sys.exit(0)
        # torch.cuda.manual_seed(args["seed"])
        # FloatTensor take much less memory on GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(args["device_id"])
    main(args)
