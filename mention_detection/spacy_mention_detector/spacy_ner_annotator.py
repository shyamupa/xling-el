import json
import logging
from typing import List, Dict, Union, Any

from ccg_nlpy.pipeline_base import PipelineBase, TextAnnotation
from spacy.tokens import Doc, Span
from ccg_nlpy import local_pipeline

from mention_detection.mention_detector import MentionDetector
from mention_detection.spacy_mention_detector.spacy_utils import getSpacyNLP, lang2spacy_model, get_tokenized_sentences, \
    getTokens


class SpacyNER_Annotator(MentionDetector):
    def __init__(self, pipeline: PipelineBase, lang: str = 'en', ignore_misc: bool = True, verbose=True):
        super().__init__()
        self.pipeline = pipeline
        self.nlp = getSpacyNLP(model_name=lang2spacy_model[lang])
        self.viewname = "MYNER" + "_{}".format(lang)
        self.viewtype = "edu.illinois.cs.cogcomp.core.datastructures.textannotation.SpanLabelView"
        self.classname = self.__class__.__name__
        self.ignore_misc = ignore_misc
        self.verbose = verbose

    def get_provided_view(self) -> str:
        return self.viewname

    def get_mentions_from_text(self, text):
        doc: Doc = self.nlp(text)

        cons_list = self.get_cons(doc=doc)
        viewData = {
            "constituents": cons_list,
            "viewType": self.viewtype,
            "viewName": self.viewname,
            "score": 1,
            "generator": self.classname
        }
        myner_view = {"viewName": self.viewname, "viewData": [viewData]}

        sentences = [[tok.text for tok in sent] for sent in get_tokenized_sentences(doc)]
        docta = self.pipeline.doc(sentences, pretokenized=True)
        ta_json = docta.as_json
        ta_json["views"].append(myner_view)
        docta.add_view(view_name=self.viewname, response=json.dumps(ta_json))
        # print(json.dumps(ta_json, indent=4, sort_keys=True))
        return ta_json

    def get_cons(self, doc: Doc) -> List[Dict[str, Union[float, Any]]]:
        """
        Creates a list of NER constituents from Spacy NER.
        :param doc:
        :return:
        """
        cons_list = []
        if self.verbose:
            logging.info("tokens:%s", list(enumerate(getTokens(doc))))
        for ent in doc.ents:
            ent: Span = ent
            sent: Span = ent.sent
            if self.ignore_misc and ent.label_ == "MISC":
                continue
            new_cons = {
                "end": ent.end,
                "label": ent.label_,
                "score": 1.0,
                "start": ent.start,
                "tokens": ent.text
            }
            if self.verbose:
                print("sent.start, sent.end", sent.start, sent.end)
                logging.info(f"{ent.text} {ent.start} {ent.end} {ent.label_}")
                logging.info("detected ner:%s", new_cons)
            cons_list.append(new_cons)
        return cons_list


if __name__ == '__main__':
    lang = "en"

    # text = "He died in the destruction of the Space Shuttle \"Challenger\", on which he was serving as Mission " \
    #        "Specialist for mission STS-51-L. "

    # SPANISH
    text = "Desde el anuncio de su campaña presidencial en febrero de 2007, Obama hizo hincapié en poner fin a la " \
           "Guerra de Irak, el aumento de la independencia energética y la prestación de asistencia sanitaria " \
           "universal como las grandes prioridades nacionales. El 10 de febrero de 2007 anunció su candidatura a la " \
           "presidencia de los Estados Unidos y el 3 de junio de 2008 se convirtió en el candidato del Partido " \
           "Demócrata. "

    # FRENCH
    text = "Ancien chauffeur de bus puis leader syndical, il est membre du Mouvement Cinquième République (MVR). " \
           "Nicolás Maduro se marie en 1988 avec Adriana Guerra Angulo, avec qui il a un fils, Nicolás Maduro Guerra. " \
           "En juillet 2013, il se remarie avec Cilia Flores."

    # GERMAN
    text = "Angela Merkel wurde im Elim-Krankenhaus im Hamburger Stadtteil Eimsbüttel als erstes Kind des " \
           "evangelischen Theologen Horst Kasner und seiner Frau Herlind Kasner. Horst Kasner hatte ab 1948 an den " \
           "Universitäten Heidelberg und Hamburg sowie an der Kirchlichen Hochschule Bethel in Bielefeld Theologie " \
           "studiert. Seine Frau war Lehrerin für Latein und Englisch. "

    # ITALIAN
    # text = "Nasce a Volturara Appula nel 1964, figlio del segretario comunale Nicola Conte e di Lillina Roberti, " \
    #        "maestra elementare. Ancora piccolo, si trasferisce con la famiglia a Candela e quindi a San " \
    #        "Giovanni Rotondo a seguito dei cambi di sede lavorativa del padre. Si diploma al Liceo Classico " \
    #        "\"Pietro Giannone\" di San Marco in Lamis."
    pipeline = local_pipeline.LocalPipeline()
    ner = SpacyNER_Annotator(lang='de', pipeline=pipeline)
    ccgdoc_dict = ner.get_mentions_from_text(text)
    ccgdoc = TextAnnotation(json.dumps(ccgdoc_dict))
    ner_cons_list = ccgdoc.get_view(ner.get_provided_view()).cons_list
    print(len(ner_cons_list))
    print([ner_cons for ner_cons in ner_cons_list])
