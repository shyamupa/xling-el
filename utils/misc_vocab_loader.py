import logging
import os
import sys

from utils.constants import NULL_TITLE_WID, OOV_TOKEN
from utils.misc_utils import load_vocab, read_candidates_dict, load, save


class VocabLoader(object):
    def __init__(self):
        self.initialize_all_dicts()

    def set_config(self, config):
        self.config = config

    def initialize_all_dicts(self):
        self.word2idx, self.idx2word = None, None
        self.type2idx, self.idx2type = None, None
        self.wid2idx, self.idx2wid = None, None
        self.wid2title, self.title2wid = None, None
        self.lang2trval_cands_dict = {}
        self.trval_cwikis = None
        self.wid2types = None
        self.test_knwen_cwikis, self.test_allen_cwikis = None, None
        self.wid2desc = None
        self.wid2descvec = None

    def load_type_vocab(self, path=None):
        if path is None:
            path = self.config.label_vocab
        if self.type2idx is None:
            if not os.path.exists(path):
                logging.fatal("type vocab pkl %s missing", path)
                sys.exit(0)
            self.type2idx, self.idx2type = load_vocab(path)
            logging.info("loaded type vocab of size %d", len(self.type2idx))
        return self.type2idx, self.idx2type

    # def load_wid2desc(self, path=None):
    #     pkl_path = path + ".pkl"
    #     if os.path.exists(pkl_path):
    #         logging.info("pkl found! loading %s", pkl_path)
    #         wid2desc = load(pkl_path)
    #     else:
    #         logging.info("loading known wids descriptions")
    #         wid2desc = load_wid2desc(path)
    #         logging.info("saving pkl wid2desc")
    #         save(pkl_path, wid2desc)
    #     self.wid2desc = map_desc(wid2desc, w2i=self.word2idx)
    #     return self.wid2desc

    def load_mix0_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.mix0_cand_dict = load(pkl_path)
        else:
            logging.info("loading test canddict")
            self.mix0_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.mix0_cand_dict)

    def load_mix1_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.mix1_cand_dict = load(pkl_path)
        else:
            logging.info("loading test canddict")
            self.mix1_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.mix1_cand_dict)

    def load_mix2_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.mix2_cand_dict = load(pkl_path)
        else:
            logging.info("loading test canddict")
            self.mix2_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.mix2_cand_dict)

    def load_mix3_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.mix3_cand_dict = load(pkl_path)
        else:
            logging.info("loading test canddict")
            self.mix3_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.mix3_cand_dict)

    def load_test_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.test_cand_dict = load(pkl_path)
        else:
            logging.info("loading test canddict")
            self.test_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.test_cand_dict)

    def load_val_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.val_cand_dict = load(pkl_path)
        else:
            logging.info("loading val canddict")
            self.val_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.val_cand_dict)

    def load_val2_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.val2_cand_dict = load(pkl_path)
        else:
            logging.info("loading val canddict")
            self.val2_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.val2_cand_dict)

    def load_val3_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.val3_cand_dict = load(pkl_path)
        else:
            logging.info("loading val canddict")
            self.val3_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.val3_cand_dict)

    def load_train_cand_dict(self, path):
        pkl_path = path + ".candict.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.trval_cand_dict = load(pkl_path)
        else:
            logging.info("loading train canddict")
            self.trval_cand_dict = read_candidates_dict(path)
            save(pkl_path, self.trval_cand_dict)

    def load_coh2idx(self, path):
        pkl_path = path + ".coh.pkl"
        if os.path.exists(pkl_path):
            logging.info("pkl found! loading %s", pkl_path)
            self.coh2idx, self.idx2coh = load(pkl_path)
        else:
            logging.info("loading coh2idx")
            self.coh2idx, self.idx2coh = {OOV_TOKEN: 0}, {0: OOV_TOKEN}
            idx = 1
            for line in open(path):
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    logging.info("bad line %s", parts)
                    continue
                cohstr, cnt = parts
                if cohstr in self.coh2idx:
                    logging.info("duplicate! %s", cohstr)
                    continue
                self.coh2idx[cohstr] = idx
                self.idx2coh[idx] = cohstr
                idx += 1
            obj = self.coh2idx, self.idx2coh
            save(pkl_path, obj)
        logging.info("coh str vocab %d", len(self.coh2idx))

    def load_word2idx_and_embeddings(self, vocab_file, embedding_path=None, norm=True):
        word2idx_pkl_path, embeddings_pkl_path, _, _ = get_word2idx_pickle_paths(embedding_path=embedding_path,
                                                                                 vocab_file=vocab_file,
                                                                                 norm=norm)
        if os.path.exists(word2idx_pkl_path) and os.path.exists(embeddings_pkl_path):
            logging.info("loading word2idx from %s", word2idx_pkl_path)
            self.word2idx, self.idx2word = load(word2idx_pkl_path)
            logging.info("loading embeddings from %s", embeddings_pkl_path)
            self.embeddings, self.oov_mask = load(embeddings_pkl_path)
            logging.warning("vocab size %d", len(self.word2idx))
            assert len(self.word2idx) == len(self.embeddings) and len(self.embeddings) == len(self.oov_mask)
        else:
            if not os.path.exists(word2idx_pkl_path):
                logging.info("%s not found", word2idx_pkl_path)
            if not os.path.exists(embeddings_pkl_path):
                logging.info("%s not found", embeddings_pkl_path)
            logging.info("run create_word2idx first! exiting.")
            sys.exit(0)

    def load_embeddings(self, embeddings_pkl_path):
        if os.path.exists(embeddings_pkl_path):
            logging.info("loading embeddings from %s", embeddings_pkl_path)
            self.embeddings, self.oov_mask = load(embeddings_pkl_path)
            logging.warning("vocab size %d", len(self.word2idx))
            assert len(self.word2idx) == len(self.embeddings) and len(self.embeddings) == len(self.oov_mask)
        else:
            if not os.path.exists(embeddings_pkl_path):
                logging.info("%s not found", embeddings_pkl_path)
            logging.info("run create_word2idx first! exiting.")
            sys.exit(0)

    def load_word2idx(self, word2idx_pkl_path=True):
        if os.path.exists(word2idx_pkl_path):
            logging.info("loading word2idx from %s", word2idx_pkl_path)
            word2idx, idx2word = load(word2idx_pkl_path)
            self.word2idx, self.idx2word = word2idx, idx2word
            logging.warning("vocab size %d", len(word2idx))
        else:
            logging.info("%s not found", word2idx_pkl_path)
            logging.info("create word2idx first! exiting.")
            sys.exit(0)

    def load_wid2idx(self, kb_file):
        # kb_file = "data/enwiki/wid_title_mid_types_counts.txt"
        pkl_path = kb_file + ".wid2idx.pkl"
        if os.path.exists(pkl_path):
            logging.info("wid2idx pkl found! loading map %s", pkl_path)
            self.wid2idx, self.idx2wid = load(pkl_path)
        else:
            self.wid2idx, self.idx2wid = {NULL_TITLE_WID: 0}, {0: NULL_TITLE_WID}
            for idx, line in enumerate(open(kb_file)):
                parts = line.strip().split("\t")
                _, wid, _, _, _ = parts
                self.wid2idx[wid] = idx + 1
                self.idx2wid[idx + 1] = wid
            obj = self.wid2idx, self.idx2wid
            save(pkl_path, obj)
        return self.wid2idx, self.idx2wid


def get_word2idx_pickle_paths(embedding_path, vocab_file, scale, norm, lower):
    if "," in embedding_path:
        embedding_paths = embedding_path.split(",")
    else:
        embedding_paths = [embedding_path]
    if "," in vocab_file:
        vocab_paths = vocab_file.split(",")
    else:
        vocab_paths = [vocab_file]
    # assert len(vocab_paths) == len(embedding_paths)

    embedding_names = [os.path.basename(path) for path in embedding_paths]
    vocab_names = [os.path.basename(path) for path in vocab_paths]
    base = "_".join(vocab_names) + "." + "_".join(embedding_names) + "." + str(norm) + "." + str(lower) + "." + str(
        scale)
    word2idx_pkl_path = "data/vocabs/" + base + ".word2idx.pkl"
    embeddings_pkl_path = "data/vocabs/" + base + ".embeddings.pkl"

    return word2idx_pkl_path, embeddings_pkl_path, vocab_paths, embedding_paths
