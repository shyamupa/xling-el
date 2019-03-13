from ccg_nlpy import remote_pipeline, local_pipeline
import json
from intervaltree import IntervalTree
import argparse
import logging

from mention_detection.mention_detector import MentionDetector
from wiki_kb.candidate_gen_v2 import CandidateGenerator
from ccg_nlpy.pipeline_base import PipelineBase


def ngrams(tokens_list, ngram_size):
    output = []
    for i in range(len(tokens_list) - ngram_size + 1):
        output.append(tokens_list[i:i + ngram_size])
    return output


class PoormanMentionDetector(MentionDetector):
    def __init__(self, pipeline: PipelineBase, cg: CandidateGenerator):
        super().__init__()
        self.cg = cg
        self.pipeline = pipeline
        self.viewname = "MYNER"
        self.viewtype = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.SpanLabelView"
        self.classname = self.__class__.__name__

    def get_mentions_from_text(self, text):
        docta = self.pipeline.doc(text)
        ta_json = docta.as_json
        # print(json.dumps(ta_json, indent=4, sort_keys=True))
        cons_list = self.get_cons(docta)
        viewData = {
            "constituents": cons_list,
            "viewType": self.viewtype,
            "viewName": self.viewname,
            "score": 1,
            "generator": self.classname
        }
        myner_view = {"viewName": self.viewname, "viewData": [viewData]}
        ta_json["views"].append(myner_view)
        docta.add_view(view_name=self.viewname, response=json.dumps(ta_json))
        print(json.dumps(ta_json, indent=4, sort_keys=True))
        return ta_json

    def get_mentions_from_file(self, doc_file):
        doctext = open(doc_file).read()
        doctext = doctext.strip()
        docta = self.pipeline.doc(doctext)
        ta_json = docta.as_json
        # print(json.dumps(ta_json, indent=4, sort_keys=True))
        cons_list = self.get_cons(docta)
        viewData = {
            "constituents": cons_list,
            "viewType": self.viewtype,
            "viewName": self.viewname,
            "score": 1,
            "generator": self.classname
        }
        myner_view = {"viewName": self.viewname, "viewData": [viewData]}
        ta_json["views"].append(myner_view)
        docta.add_view(view_name=self.viewname, response=json.dumps(ta_json))
        # print(json.dumps(ta_json, indent=4, sort_keys=True))
        return ta_json
        # print(self.ccgdoc.get_views)

    def get_cons(self, docta):
        cons_list = []
        tokens_list = list(enumerate(docta.get_tokens))
        spans_so_far = IntervalTree()
        for ngram_size in [4, 3, 2, 1]:
            for ngram in ngrams(tokens_list=tokens_list, ngram_size=ngram_size):
                ngram_start = ngram[0][0]
                ngram_end = ngram[-1][0] + 1
                ngram_string = " ".join([n[1] for n in ngram])
                # print(ngram, ngram_start, ngram_end, ngram_string)
                cands = self.cg.get_candidates(ngram_string, pretokenized=True)
                logging.info("query: %s", ngram_string)
                logging.info("cands found: %d", len(cands))
                if len(cands) == 0:
                    continue
                most_prob_cand = cands[0]
                new_cons = {
                    "end": ngram_end,
                    "label": "MENTION",
                    "score": 1.0,
                    "start": ngram_start,
                    "most_prob_cand": most_prob_cand.en_title,
                    "most_prob_prob": most_prob_cand.p_t_given_s,
                    "ncands": len(cands),
                    "tokens": ngram_string
                }
                overlap_mentions = spans_so_far.overlap(begin=ngram_start, end=ngram_end)
                if len(overlap_mentions) > 0:
                    # do not allow overlapping/nested mentions
                    continue
                else:
                    spans_so_far.addi(begin=ngram_start, end=ngram_end)
                    cons_list.append(new_cons)
        logging.info("#mentions found:%d", len(cons_list))
        logging.info("#total tokens:%d", len(tokens_list))
        return cons_list


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
                                 fallback=False, debug=True)
    wiki_cg.load_probs("data/{}wiki/probmap/{}wiki-{}".format(args["lang"], args["lang"], args["date"]))
    # pipeline = local_pipeline.LocalPipeline()
    pipeline = remote_pipeline.RemotePipeline(server_api='http://macniece.seas.upenn.edu:4001')
    md = PoormanMentionDetector(cg=wiki_cg, pipeline=pipeline)
    md.get_mentions_from_file("testdoc.txt")
