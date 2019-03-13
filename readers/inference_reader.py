import copy
import logging
import time

import numpy as np
import utils.constants as K

from datastructs.mention import Mention
from mention_detection.mention_detector import MentionDetector
from mention_detection.spacy_mention_detector.spacy_ner_annotator import SpacyNER_Annotator
from readers.mention_reader import MentionReader, pad_batch
from utils.arguments import PARSER
from ccg_nlpy import remote_pipeline, local_pipeline

from utils.misc_vocab_loader import VocabLoader
# from readers.training_reader import DataReader
# from mention_detection.poorman_mention_detector import PoormanMentionDetector
from wiki_kb.candidate_gen_v2 import CandidateGenerator
from ccg_nlpy import TextAnnotation
import json

from wiki_kb.title_normalizer_v2 import TitleNormalizer


class InferenceReader(MentionReader):
    """
    Based on Nitish's Neural-EL Reader
    """

    def __init__(self, args, mention_detector: MentionDetector, cg: CandidateGenerator, loader: VocabLoader,
                 num_cands: int, batch_size: int, strict_context=True, usecoh=True, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.typeOfReader = "inference"
        self.usecoh = usecoh
        self.mention_detector = mention_detector
        self.ner_view_name = mention_detector.get_provided_view()
        self.cg = cg
        self.num_cands = num_cands
        self.batch_size = batch_size
        self.shuffle = False
        self.en_normalizer = TitleNormalizer(lang="en")
        self.wid2idx, self.idx2wid = loader.wid2idx, loader.idx2wid

        # Word Vocab
        self.word2idx, self.idx2word = loader.word2idx, loader.idx2word
        self.num_words = len(self.idx2word)

        # Coherence String Vocab
        if args["usecoh"]:
            self.coh2idx, self.idx2coh = loader.coh2idx, loader.idx2coh
            self.num_coh = args["num_coh"]

        # Known WID Vocab
        self.wid2idx, self.idx2wid = loader.wid2idx, loader.idx2wid

        # Word Emb
        self.word_embedding = loader.embeddings

        self.batch_size = batch_size
        self.num_cands = num_cands
        self.strict_context = strict_context

    def set_test_doc(self, ccgdoc: TextAnnotation):
        if self.verbose:
            print("[#] Loading test text and preprocessing ... ")
        self.process_test_doc(ccgdoc=ccgdoc)
        self.mention_lines = self.create_mention_lines()
        self.mentions = []
        for line in self.mention_lines:
            m = Mention(line)
            self.mentions.append(m)

        self.men_idx = 0
        self.num_mens = len(self.mentions)
        self.epochs = 0
        self.finished = False
        if self.verbose:
            print("Test Mentions : {}".format(self.num_mens))

    def get_vector(self, word: str):
        if word in self.word_embedding:
            return self.word_embedding[word]
        else:
            return self.word_embedding['unk']

    def embed_batch(self, batch):
        batch = np.array(batch, dtype=np.int32)  # this is needed
        output_batch = self.word_embedding[batch]
        return output_batch

    def reset_test(self):
        self.men_idx = 0
        self.epochs = 0
        self.finished = False

    def process_test_doc(self, ccgdoc):
        # List of tokens
        doc_tokens = ccgdoc.get_tokens
        # sent_end_token_indices : contains index for the starting of the
        # next sentence.
        sent_end_token_indices = \
            ccgdoc.get_sentence_end_token_indices
        # print("sent_end_token_indices", sent_end_token_indices)
        # List of tokenized sentences
        self.tokenized_sentences = []
        for i in range(0, len(sent_end_token_indices)):
            start = sent_end_token_indices[i - 1] if i != 0 else 0
            end = sent_end_token_indices[i]
            sent_tokens = doc_tokens[start:end]
            self.tokenized_sentences.append(sent_tokens)
        # print("self.tokenized_sentences", self.tokenized_sentences)
        # List of ner dicts from ccg pipeline
        ner_cons_list = []
        try:
            ner_cons_list = copy.deepcopy(ccgdoc.get_view(self.ner_view_name).cons_list)
        except:
            print("NO NAMED ENTITIES IN THE DOC. EXITING")

        # SentIdx : [(tokenized_sent, ner_dict)]
        self.sentidx2ners = {}
        for ner in ner_cons_list:
            found = False
            # idx = sentIdx, j = sentEndTokenIdx
            for idx, j in enumerate(sent_end_token_indices):
                sent_start_token = sent_end_token_indices[idx - 1] \
                    if idx != 0 else 0
                # ner['end'] is the idx of the token after ner
                if ner['end'] < j:
                    if idx not in self.sentidx2ners:
                        self.sentidx2ners[idx] = []
                    ner['start'] = ner['start'] - sent_start_token
                    ner['end'] = ner['end'] - sent_start_token - 1
                    self.sentidx2ners[idx].append(
                        (self.tokenized_sentences[idx], ner))
                    found = True
                if found:
                    break

    def create_mention_lines(self):
        """Convert NERs from document to list of mention strings"""
        mentions = []
        # Make Document Context String for whole document
        cohStr = ""
        for sent_idx, s_nerDicts in self.sentidx2ners.items():
            for s, ner in s_nerDicts:
                cohStr += ner['tokens'].replace(' ', '_') + ' '

        cohStr = cohStr.strip()

        for idx, sent_tokens in enumerate(self.tokenized_sentences):
            if idx in self.sentidx2ners:
                s_nerDicts = self.sentidx2ners[idx]
                for s, ner in s_nerDicts:
                    mention = "%s\t%s\t%s" % ("unk_mid", "unk_wid", "unkWT")
                    mention = mention + '\t' + str(ner['start'])
                    mention = mention + '\t' + str(ner['end'])
                    mention = mention + '\t' + str(ner['tokens'])
                    mention = mention + '\t' + ' '.join(sent_tokens)
                    mention = mention + '\t' + "UNK_TYPES"
                    mention = mention + '\t' + cohStr
                    mentions.append(mention)
        return mentions

    # def bracketMentionInSentence(self, s, nerDict):
    #     tokens = s.split(" ")
    #     start = nerDict['start']
    #     end = nerDict['end']
    #     tokens.insert(start, '[[')
    #     tokens.insert(end + 2, ']]')
    #     return ' '.join(tokens)

    def _read_mention(self):
        if self.men_idx >= len(self.mentions):
            return None
        mention = self.mentions[self.men_idx]
        self.men_idx += 1
        # if self.verbose:
        #     print(f"mention: {self.men_idx} total: {self.num_mens} epoch:{self.epochs}")
        if self.men_idx == self.num_mens:
            # self.men_idx = 0
            self.epochs += 1
        return mention

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
            # assert m.idx_version is True

            # rv = self.get_candidate_batch(m)
            rv = self.make_candidates_cprobs(m)
            if rv is None:
                continue

            wids, wid_cprobs = rv

            # Left and Right context
            left_toks, right_toks = self.get_mention_contexts(m)
            if len(left_toks) == 0 or len(right_toks) == 0:
                logging.info("empty left or right cxt")
                continue
            left_idxs, right_idxs = [self.convert_word2idx(tok) for tok in left_toks], [self.convert_word2idx(tok) for
                                                                                        tok in right_toks]
            left_batch.append(left_idxs)
            right_batch.append(right_idxs)

            if self.usecoh:
                coh_idxs = self.make_coherence_batch(m=m)
                coherence_batch.append(coh_idxs)

            cand_wid_idxs_batch.append(wids)
            cand_wid_cprobs_batch.append(wid_cprobs)

        return (left_batch, right_batch, doc_bow_batch, gold_wid_desc_batch, types_batch,
                coherence_batch, cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch)

    def make_coherence_batch(self, m):
        idxs = []
        for coh_str in m.coherence:
            # idxs.append(coh_str)
            if coh_str in self.coh2idx:
                idxs.append(self.coh2idx[coh_str])
        if not idxs:
            idxs.append(self.coh2idx[K.OOV_TOKEN])
            # idxs.append(K.OOV_ID)
        return idxs

    @staticmethod
    def get_mention_contexts(mention):
        start, end = mention.start_token, mention.end_token
        # Context inclusive of mention surface
        left_idxs = mention.sent_tokens[0:end + 1]
        right_idxs = mention.sent_tokens[start:][::-1]
        return left_idxs, right_idxs

    def make_candidates_cprobs(self, m):
        surface = m.surface.lower()
        wiki_titles, wids, wid_cprobs = self.extract_cands(self.cg.get_candidates(surface=surface))
        if self.verbose:
            for title, wid, cprob in zip(wiki_titles, wids, wid_cprobs):
                print(title, wid, cprob)
        # Mapped to idxs in wid2idx already
        wids = list(map(lambda wid: self.wid2idx[wid], wids))
        if len(wids) < self.num_cands:
            wids += [K.NULL_TITLE_ID] * (self.num_cands - len(wids))
            wid_cprobs += [0.0] * (self.num_cands - len(wid_cprobs))
        assert len(wids) == len(wid_cprobs)
        return wids, wid_cprobs

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
        doc_bow_batch = []

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

    def convert_word2idx(self, word: str) -> int:
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[K.OOV_TOKEN]

    def next_test_batch(self):
        return self._next_padded_batch()

    def extract_cands(self, cands):
        wiki_titles, wids, wid_cprobs = [], [], []
        for cand in cands:
            wikititle, p_t_given_s, p_s_given_t = cand.en_title, cand.p_t_given_s, cand.p_s_given_t
            nrm_title = self.en_normalizer.normalize(wikititle)
            if nrm_title == K.NULL_TITLE:  # REMOVED or nrm_title not in en_normalizer.title2id
                logging.info("bad cand %s nrm=%s", wikititle, nrm_title)
                continue
            wiki_id = self.en_normalizer.title2id[nrm_title]
            wiki_titles.append(nrm_title)
            wids.append(wiki_id)
            wid_cprobs.append(p_t_given_s)
        return wiki_titles, wids, wid_cprobs


if __name__ == '__main__':
    sttime = time.time()
    args = PARSER.parse_args()
    args = vars(args)
    loader = VocabLoader()
    loader.load_word2idx(word2idx_pkl_path=args["vocabpkl"])
    loader.load_embeddings(embeddings_pkl_path=args["vecpkl"])
    loader.load_wid2idx(kb_file=args["kb_file"])
    args["num_entities"] = len(loader.wid2idx)
    args["num_words"] = len(loader.word2idx)

    wiki_cg = CandidateGenerator(kbfile=args["kb_file"], K=args["ncands"], lang=args["lang"],
                                 fallback=False, debug=True)
    wiki_cg.load_probs("data/{}wiki/probmap/{}wiki-{}".format(args["lang"], args["lang"], args["date"]))

    # pipeline = remote_pipeline.RemotePipeline(server_api='http://macniece.seas.upenn.edu:4001')
    pipeline = local_pipeline.LocalPipeline()

    md = SpacyNER_Annotator(lang=args["lang"], pipeline=pipeline)
    # md = PoormanMentionDetector(cg=wiki_cg, pipeline=pipeline)

    inf_reader = InferenceReader(args=args,
                                 mention_detector=md,
                                 cg=wiki_cg,
                                 batch_size=args["batch_size"],
                                 loader=loader,
                                 num_cands=args["ncands"],
                                 strict_context=False,
                                 usecoh=args["usecoh"])

    test_file = args["test_doc"]
    # print("[#] Test Mentions File : {}".format(test_file))
    with open(test_file, 'r') as f:
        lines = f.read().strip().split("\n")
    assert len(lines) == 1, "Only support inference for single doc"
    doctext = lines[0].strip()

    ccgdoc_dict = inf_reader.mention_detector.get_mentions_from_text(doctext)
    ccgdoc = TextAnnotation(json.dumps(ccgdoc_dict))

    inf_reader.set_test_doc(ccgdoc=ccgdoc)
