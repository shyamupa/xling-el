import copy
import json
from typing import List

from ccg_nlpy import local_pipeline, TextAnnotation
from ccg_nlpy.pipeline_base import PipelineBase

from mention_detection.spacy_mention_detector.spacy_ner_annotator import SpacyNER_Annotator
from mention_detection.stanfordnlp.stanfordnlp_mention_detector import StanfordNLPMentionDetector
from model.my_model import MyModel
from readers.inference_reader import InferenceReader
from utils.acc_evaluator import compute_joint_probs
from utils.arguments import PARSER
from utils.el_runner import ELRunner
from utils.misc_vocab_loader import VocabLoader
from wiki_kb.candidate_gen_v2 import CandidateGenerator
import torch.nn.functional as F
from ccg_nlpy.server.annotator import Annotator
import utils.constants as K
import logging

logging.basicConfig(format=':%(levelname)s: %(message)s', level=logging.INFO)


def prune_multiple_spaces(sentence: str):
    """ Prune multiple spaces in a sentence and replace with single space
    Parameters:
    -----------
    sentence: Sentence string with mulitple spaces

    Returns:
    --------
    cleaned_sentence: String with only single spaces.
    """

    sentence = sentence.strip()
    tokens = sentence.split(' ')
    tokens = [t for t in tokens if t != '']
    if len(tokens) == 1:
        return tokens[0]
    else:
        return ' '.join(tokens)


class XELAnnotator(Annotator):
    def __init__(self, args, inf_reader: InferenceReader, pipeline: PipelineBase, viewname_prefix="NEURAL_XEL",
                 verbose: bool = True):
        # logging.info(args)
        self.lang = args["lang"]
        provided_view = viewname_prefix + "_{}".format(self.lang)
        # We do our own NER
        required_views = []
        super().__init__(pipeline, provided_view, required_views)
        self.model = MyModel(args=args)
        self.restore_path = args["restore"]
        self.ncands = args["ncands"]
        self.inf_reader = inf_reader
        self.params_loaded = False
        self.verbose = verbose

    def load_params(self):
        if self.params_loaded is not True:
            ELRunner.load_checkpoint(model=self.model, optimizer=None, ckpt_path=self.restore_path)
            self.params_loaded = True
        else:
            logging.info("params already loaded")

    def get_text_annotation_for_model(self, text: str, required_views: List[str]) -> TextAnnotation:
        # TODO This is a problem with ccg_nlpy text annotation, it does not like newlines (e.g., marking paragraphs)
        text = text.replace("\n", "")
        pretokenized_text = [text.split(" ")]
        required_views = ",".join(required_views)
        ta_json = self.pipeline.call_server_pretokenized(pretokenized_text=pretokenized_text, views=required_views)
        ta = TextAnnotation(json_str=ta_json)
        return ta

    def inference_on_text(self, text: str) -> TextAnnotation:
        ccgdoc_dict = self.inf_reader.mention_detector.get_mentions_from_text(text)
        ccgdoc = TextAnnotation(json.dumps(ccgdoc_dict))
        # return self.inference_on_ta(ccgdoc)
        wiki_view = self.create_view(ccgdoc)
        wiki_view.view_name = self.provided_view
        ccgdoc.view_dictionary[self.provided_view] = wiki_view
        return ccgdoc

    def add_view(self, docta: TextAnnotation) -> TextAnnotation:
        text = docta.text
        # TODO: This needs fixing in apelles. It appends a newline to each line.
        text = text.replace("\n", " ")
        # TODO this step is not ideal. it discards any previously added views in the docta
        return self.inference_on_text(text=text)

    def run_model_inference(self, docta: TextAnnotation):
        self.inf_reader.set_test_doc(ccgdoc=docta)
        test_iterator = self.inf_reader
        self.model.eval()
        all_wid_idxs = []
        all_cxt_probs = []
        all_prior_probs = []
        for bid, batch in enumerate(test_iterator):
            cand_wid_idxs_batch, cand_wid_cprobs_batch, nocands_mask_batch = batch[-3:]
            if len(cand_wid_cprobs_batch) < test_iterator.batch_size:
                self.model.batch_size = len(cand_wid_cprobs_batch)
            # nb x ncands
            batch = self.model.prepare_batch(batch, istest=True)
            logits = self.model.forward(batch)
            cxt_logits = logits["cxt_logits"]
            cxt_probs = F.softmax(cxt_logits)
            all_prior_probs.extend(cand_wid_cprobs_batch.tolist())
            all_wid_idxs.extend(cand_wid_idxs_batch.tolist())
            model_probs = cxt_probs.data.numpy()
            all_cxt_probs += model_probs.tolist()
        all_joint_probs = compute_joint_probs(cxt_probs=all_cxt_probs, cand_probs=all_prior_probs)
        return all_wid_idxs, all_prior_probs, all_cxt_probs, all_joint_probs

    def create_view(self, docta):

        all_wid_idxs, all_prior_probs, all_cxt_probs, all_joint_probs = self.run_model_inference(docta)

        wiki_view = copy.deepcopy(docta.get_view("MYNER" + "_{}".format(self.lang)))
        el_cons_list = wiki_view.cons_list
        numMentionsInference = len(all_wid_idxs)

        if self.verbose:
            print(f"Number of mentions in model: {len(all_wid_idxs)}")
            print(f"Number of NER mention: {len(el_cons_list)}")

        assert len(el_cons_list) == numMentionsInference

        for _, m_info in enumerate(zip(el_cons_list, all_wid_idxs, all_prior_probs, all_cxt_probs, all_joint_probs)):
            ner_cons, wididxs, priors, cxts, joints = m_info
            priorScoreMap = {}
            contextScoreMap = {}
            jointScoreMap = {}

            maxJointProb = 0.0
            maxJointEntity = ""
            for (wid_idx, prior_score, cxt_score, joint_score) in zip(wididxs, priors, cxts, joints):
                wiki_title = self.inf_reader.idx2wid[wid_idx]
                priorScoreMap[wiki_title] = prior_score
                contextScoreMap[wiki_title] = cxt_score
                jointScoreMap[wiki_title] = joint_score

                if joint_score > maxJointProb:
                    maxJointProb = joint_score
                    maxJointEntity = wiki_title

            if self.verbose:
                ner_cons["jointScoreMap"] = jointScoreMap
                ner_cons["contextScoreMap"] = contextScoreMap
                ner_cons["priorScoreMap"] = priorScoreMap

            # add max scoring entity as label
            final_title = self.inf_reader.en_normalizer.get_id2title(wid=maxJointEntity)
            if final_title is not K.NULL_TITLE:
                # final_label = f"<a href=https://en.wikipedia.org/wiki/{final_title}>{final_title}</a>"
                final_label = f"en.wikipedia.org/wiki/{final_title}"
            else:
                final_label = final_title
            ner_cons["label"] = final_label
            ner_cons["score"] = maxJointProb
            if self.verbose:
                print(ner_cons)

        return wiki_view


def setup_annotator(args, pipeline):
    loader = VocabLoader()
    loader.load_word2idx(word2idx_pkl_path=args["vocabpkl"])
    loader.load_embeddings(embeddings_pkl_path=args["vecpkl"])
    loader.load_wid2idx(kb_file=args["kb_file"])
    args["num_entities"] = len(loader.wid2idx)
    args["num_words"] = len(loader.word2idx)
    logging.info("num_entities %d", args["num_entities"])
    args["filter_sizes"] = list(map(int, args["filter_sizes"].split(",")))
    if args["usecoh"]:
        loader.load_coh2idx(path=args["cohstr"])
        args["num_coh"] = len(loader.coh2idx)

    wiki_cg = CandidateGenerator(kbfile=args["kb_file"], K=args["ncands"], lang=args["lang"],
                                 fallback=False, debug=True)
    wiki_cg.load_probs("data/{}wiki/probmap/{}wiki-{}".format(args["lang"], args["lang"], args["date"]))
    # pipeline = remote_pipeline.RemotePipeline(server_api='http://macniece.seas.upenn.edu:4001')
    if args["lang"] in ["de", "es", "fr", "it"]:
        md = SpacyNER_Annotator(lang=args["lang"], pipeline=pipeline, verbose=args["verbose"])
    elif args["lang"] in ["zh"]:
        md = StanfordNLPMentionDetector(lang=args["lang"], pipeline=pipeline, verbose=args["verbose"])
    else:
        raise NotImplementedError
    inf_reader = InferenceReader(args=args, mention_detector=md, cg=wiki_cg,
                                 batch_size=args["batch_size"],
                                 loader=loader,
                                 num_cands=args["ncands"],
                                 strict_context=False,
                                 usecoh=args["usecoh"],
                                 verbose=args["verbose"])
    annotator = XELAnnotator(pipeline=pipeline, args=args, inf_reader=inf_reader, verbose=args["verbose"])
    return annotator


def main(args):
    pipeline = local_pipeline.LocalPipeline()
    annotator = setup_annotator(args, pipeline=pipeline)

    test_file = args["test_doc"]
    out_file = args["out_doc"]
    print("[#] Test Mentions File : {}".format(test_file))
    with open(test_file, 'r') as f:
        lines = f.read().strip().split("\n")
    assert len(lines) == 1, "Only support inference for single doc"
    doctext = lines[0].strip()

    ta = annotator.inference_on_text(text=doctext)
    ta_json = ta.as_json
    json.dump(ta_json, open(out_file, "w"), indent=True)


if __name__ == '__main__':
    args = PARSER.parse_args()
    args = vars(args)
    main(args)
