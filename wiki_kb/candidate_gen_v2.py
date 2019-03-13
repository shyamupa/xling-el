# coding=utf-8
from __future__ import print_function
from __future__ import division

import argparse
import logging

from wiki_kb.title_normalizer_v2 import TitleNormalizer
from wiki_kb.candidate_utils import combine_duplicates_n_sort
from utils.constants import NULL_TITLE
from utils.misc_utils import load_nekb, load_langlinks_mongo, load_prob_map_mongo

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)
from utils.text_utils import tokenizer
from wiki_kb.candidate import Candidate
from hanziconv import HanziConv

__author__ = 'Shyam'


class CandidateGenerator:
    def __init__(self, lang, date="20170520", K=10, fallback=True, use_eng=False, kbfile=None, debug=False):
        self.lang = lang
        self.date = date
        if self.lang != "en":
            self.fr2entitles, _ = load_langlinks_mongo(lang)
        else:
            self.fr2entitles = None
        self.K = K
        self.use_eng = use_eng
        if kbfile is not None:
            self.wid2ne, self.mid2ne, self.title2ne = load_nekb(kbfile)
        else:
            logging.info("No KB given ... all of wikipedia will be used")
            self.wid2ne, self.mid2ne, self.title2ne = None, None, None
        self.no_cand_after_pruning = 0
        self.oov_surface_tokens = 0
        self.fallback = fallback  # fallback on word level cand gen if phrase level does not give much.
        self.debug = debug
        if self.fallback:
            logging.info("will fallback to word level")

    def load_probs(self, out_prefix):
        self.p2t2prob = load_prob_map_mongo(out_prefix, "p2t2prob")
        # self.t2p2prob = load_prob_map(out_prefix, "t2p2prob")
        if self.fallback:
            self.w2t2prob = load_prob_map_mongo(out_prefix, "w2t2prob")
            # self.t2w2prob = load_prob_map(out_prefix, "t2w2prob")
        if self.use_eng:
            logging.info("also using ENG probs")
            self.en_p2t2prob = load_prob_map_mongo("data/enwiki/probmap/enwiki-{}".format(self.date), "p2t2prob")
            self.en_w2t2prob = load_prob_map_mongo("data/enwiki/probmap/enwiki-{}".format(self.date), "w2t2prob")

    def get_candidates(self, surface, pretokenized=False):
        cands = self.__get_candidate_by_phrase(lang=self.lang,
                                               K=self.K,
                                               surface=surface,
                                               p2t2prob=self.p2t2prob)
        if self.lang == "zh":
            if "•" in surface:
                surface = surface.replace("•", "·")
            if "．" in surface:
                surface = surface.replace("．", "·")
            surface = HanziConv.toSimplified(surface)
            # logging.info("zh surface %s",surface)

        if self.lang == "zh" and len(cands) == 0:
            # logging.info("Nothing for surface %s", surface)
            surface = surface.replace(" ", "")
            # logging.info("new surface %s", surface)
            cands += self.__get_candidate_by_phrase(lang=self.lang,
                                                    K=self.K,
                                                    surface=surface,
                                                    p2t2prob=self.p2t2prob)

        if len(cands) == 0 and self.use_eng:
            # logging.info("trying ENG probs for %s", surface)
            cands += self.__get_candidate_by_phrase(lang="en",
                                                    K=self.K,
                                                    surface=surface,
                                                    p2t2prob=self.en_p2t2prob)
        if not self.fallback:
            # Duplicates can come due to fr2en mapping (fr1 --> en <-- fr2)
            cands = combine_duplicates_n_sort(cands)
            return cands
        if len(cands) == 0:
            cands += self.__get_candidate_by_word(lang=self.lang,
                                                  K=self.K,
                                                  surface=surface,
                                                  w2t2prob=self.w2t2prob,
                                                  pretokenized=pretokenized)
        cands = combine_duplicates_n_sort(cands)
        return cands

    def __get_candidate_by_phrase(self, lang, K, surface, p2t2prob, debug=False):
        surface = surface.lower()  # We have to do this, to reduce sparsity
        cand_list = []
        if surface in p2t2prob:
            all_titles = sorted(p2t2prob[surface].items(), key=lambda x: -x[1])
            all_titles = filter_by_kb(fr2entitles=self.fr2entitles,
                                      title2ne=self.title2ne,
                                      lang=lang,
                                      all_titles=all_titles)  # this returns (t, entitle, prob)
            top_k = all_titles[:K]
            for fr_title, en_title, prob in top_k:
                # For english fr_title = en_title
                c = Candidate(surface=surface,
                              en_title=en_title,
                              fr_title=fr_title,
                              is_gold=0,
                              p_t_given_s=p2t2prob[surface][fr_title],
                              lang=lang,
                              src="phrase")
                cand_list.append(c)
        else:
            if debug:
                logging.info("no phrase key for %s", surface)
        # logging.info("phrase gen sending %s", len(cand_list))
        return cand_list

    def __get_candidate_by_word(self, K, lang, surface, w2t2prob, pretokenized=False):
        # TODO
        surface = surface.lower()
        cand_list = []
        try:
            if pretokenized and lang == "zh":
                tokens = surface.split("·")
            elif lang != "zh":
                tokens = surface.split(" ")
            else:
                tokens = tokenizer(surface, lang)
            word_limit = K // len(tokens)
            for token in tokens:
                if token in w2t2prob:
                    all_titles = sorted(w2t2prob[token].items(), key=lambda x: -x[1])
                    all_titles = filter_by_kb(fr2entitles=self.fr2entitles,
                                              title2ne=self.title2ne,
                                              lang=lang,
                                              all_titles=all_titles)
                    top_k_for_word = all_titles[:word_limit]
                    # top_k_for_word = sorted(top_k_for_word, key=lambda x: -x[1])
                    for fr_title, en_title, prob in top_k_for_word:
                        c = Candidate(surface=surface,
                                      en_title=en_title,
                                      fr_title=fr_title,
                                      is_gold=0,
                                      p_t_given_s=w2t2prob[token][fr_title],
                                      lang=lang, src="word")
                        cand_list.append(c)
                else:
                    pass
                    # logging.info("no word key for %s", token)
            cand_list = sorted(cand_list, key=lambda x: -1.0 * x.p_t_given_s)[:self.K]
        except ValueError as e:
            logging.info("Exception on tokenizing %s", surface)
            logging.info("Exception was %s", e)
        return cand_list


def filter_by_kb(fr2entitles, title2ne, lang, all_titles):
    # TODO this sends back null titles in English. That should not happen!
    new_titles = []
    new_normalizer = 0.0
    for title, prob in all_titles:
        if lang != "en":
            if title in fr2entitles:
                en_title = fr2entitles[title]
            else:
                en_title = NULL_TITLE
        else:
            # if lang is en, en_title is title
            en_title = title

        # not in KB
        if title2ne is not None and en_title not in title2ne:
            continue

        new_titles.append((title, en_title, prob))
        new_normalizer += prob

    new_titles = [(t, en_t, p / new_normalizer) for (t, en_t, p) in new_titles]
    # print("sending", len(new_titles))
    return new_titles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='entity linker')
    parser.add_argument('--lang', type=str, required=True, help='lang code')
    parser.add_argument('--kbfile', type=str, default=None, help='limit candidates to a kb, otherwise its all of '
                                                                 'Wikipedia')
    parser.add_argument('--date', type=str, default="20170520", help='wikidump date')
    parser.add_argument('--numcands', type=int, default=10, help='max # of cands')
    parser.add_argument('--nofallback', action="store_true", help='whether to fallback to word level cand gen or not')
    parser.add_argument('--interactive', action="store_true", help='interactive candgen mode for debug')
    args = parser.parse_args()
    args = vars(args)
    wiki_cg = CandidateGenerator(kbfile=args["kbfile"], K=args["numcands"], lang=args["lang"],
                                 fallback=not args["nofallback"], debug=True)
    wiki_cg.load_probs("data/{}wiki/probmap/{}wiki-{}".format(args["lang"], args["lang"], args["date"]))
    title_normalizer = TitleNormalizer(lang="en")

    if args["interactive"]:
        try:
            while True:
                surface = input("enter surface:")
                cands = wiki_cg.get_candidates(surface, pretokenized=True)
                print("cands found:", len(cands))
                for idx, cand in enumerate(cands):
                    nrm = title_normalizer.normalize(cand.en_title)
                    print(idx, cand.en_title, cand.p_t_given_s, cand.p_s_given_t, "nrm", nrm, cand.src)
                    if nrm != cand.en_title:
                        logging.info("no match nrm %s title %s", nrm, cand.en_title)
        except KeyboardInterrupt:
            print('interrupted!')

