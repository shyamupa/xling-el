import json
import logging
import os

from ccg_nlpy import local_pipeline, TextAnnotation
# from stanfordnlp.server import CoreNLPClient
from mention_detection.stanfordnlp.stanford_client import CoreNLPClient
#
from mention_detection.mention_detector import MentionDetector


class StanfordNLPMentionDetector(MentionDetector):
    def get_provided_view(self) -> str:
        return self.viewname

    def __init__(self, pipeline, lang, verbose:bool = False):
        super().__init__()
        self.pipeline = pipeline
        self.viewname = "MYNER" + "_{}".format(lang)
        self.viewtype = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.SpanLabelView"
        self.classname = self.__class__.__name__
        self.verbose = verbose
        if 'CORENLP_HOME' not in os.environ:
            logging.info("Setting CORENLP HOME ....")
            os.environ['CORENLP_HOME'] = "/home/upadhya3/stanford_jars/stanford-corenlp-full-2018-10-05/"
        properties = {
            # segment
            "tokenize.language": "zh",
            "segment.model": "edu/stanford/nlp/models/segmenter/chinese/ctb.gz",
            "segment.sighanCorporaDict": "edu/stanford/nlp/models/segmenter/chinese",
            "segment.serDictionary": "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz",
            "segment.sighanPostProcessing": "true",
            # sentence split
            "ssplit.boundaryTokenRegex": "[.。]|[!?！？]+",
            # pos
            "pos.model": "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger",
            # ner
            "ner.language": "chinese",
            "ner.model": "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz",
            "ner.applyNumericClassifiers": "true",
            "ner.useSUTime": "false",
            # regexner
            "ner.fine.regexner.mapping": "edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab",
            "ner.fine.regexner.noDefaultOverwriteLabels": "CITY,COUNTRY,STATE_OR_PROVINCE"
        }
        annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner']
        # set up the client
        self.corenlp_client = CoreNLPClient(properties=properties,
                                            annotators=annotators,
                                            timeout=60000, memory='16G',
                                            output_format="json",
                                            be_quiet=not self.verbose)

    def get_cons(self, ann):
        cons_list = []
        for sent in ann["sentences"]:
            for ent in sent["entitymentions"]:
                new_cons = {
                    "end": ent["docTokenEnd"],
                    "label": ent["ner"],
                    "score": 1.0,
                    "start": ent["docTokenBegin"],
                    "tokens": ent["text"]
                }
                cons_list.append(new_cons)
        return cons_list

    def get_mentions_from_text(self, text):
        # with self.corenlp_client as client:
        ann = self.corenlp_client.annotate(text)
        # print(json.dumps(ann, indent=4, sort_keys=True))
        cons_list = self.get_cons(ann=ann)
        viewData = {
            "constituents": cons_list,
            "viewType": self.viewtype,
            "viewName": self.viewname,
            "score": 1,
            "generator": self.classname
        }
        myner_view = {"viewName": self.viewname, "viewData": [viewData]}

        sentences = [[token["word"] for token in sent["tokens"]] for sent in ann["sentences"]]
        docta = self.pipeline.doc(sentences, pretokenized=True)
        ta_json = docta.as_json
        ta_json["views"].append(myner_view)
        docta.add_view(view_name=self.viewname, response=json.dumps(ta_json))
        # print(json.dumps(ta_json, indent=4, sort_keys=True))
        return ta_json

    def __del__(self):
        print("stopping corenlp server ...")
        self.corenlp_client.stop()


if __name__ == '__main__':
    # example text
    print('---')
    print('input text')
    print('')

    # text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
    text = "奧巴馬的母親斯坦利·安·鄧納姆，在1942年11月29日，生於堪薩斯州威奇托圣方濟各医院，主要是英國血統。他的父親老巴拉克·奧巴馬，在1936年6月18" \
           "日，生於東非肯尼亞西部維多利亞湖邊夏亞郡科蓋若村，盧歐族人，肯亞政治家、多國政府顧問，也是學者 "

    # set up the client
    print('---')
    print('starting up Java Stanford CoreNLP Server...')
    pipeline = local_pipeline.LocalPipeline()
    md = StanfordNLPMentionDetector(pipeline=pipeline, lang="zh")
    md.get_mentions_from_text(text=text)
    ccgdoc_dict = md.get_mentions_from_text(text)
    ccgdoc = TextAnnotation(json.dumps(ccgdoc_dict))
    ner_cons_list = ccgdoc.get_view(md.get_provided_view()).cons_list
    print(len(ner_cons_list))
    print([ner_cons for ner_cons in ner_cons_list])

    text = "青年時期，奧巴馬因為自己的多種族背景，很難取得社會認同，十分自卑。十幾歲的他成了癮君子，他和任何絕望的黑人青年一樣，不知道生命的意義何在。家境貧窮，膚色經常遭人嘲笑，前途無望，成功的道路曲折得連路都找不著。他過了一段荒唐的日子，做了很多愚蠢的事，比如翹課、吸毒、泡妞等，成了不折不扣的“迷途叛逆少年”，曾以吸食大麻和可卡因來“將‘我是誰’的問題擠出腦袋”[7]。有媒體撰文認為，給青年的他帶來深刻影響的不是他的父母親，而是他的外祖父斯坦利·埃默·鄧漢姆和外祖母斯坦利·安·鄧漢姆[8]；媒體同時還披露著名黑人詩人、記者和美國共產黨、左翼活動家法蘭克·米歇爾·大衛斯也是深刻影響青年奧巴馬的人物，1960年代大衛斯就成為奧巴馬家裡的常客 "
    # print(text)
    ccgdoc_dict = md.get_mentions_from_text(text)
    ccgdoc = TextAnnotation(json.dumps(ccgdoc_dict))
    ner_cons_list = ccgdoc.get_view(md.get_provided_view()).cons_list
    print(len(ner_cons_list))
    print([ner_cons for ner_cons in ner_cons_list])
