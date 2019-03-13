from typing import List

from ccg_nlpy import local_pipeline
from ccg_nlpy.server.annotator import Annotator
from ccg_nlpy.server.multi_annotator import MultiAnnotator
from flask import Flask
from flask_cors import CORS

from readers.xel_annotator import setup_annotator
from utils.arguments import PARSER

app = Flask(__name__)
# necessary for testing on localhost
CORS(app)


def main(args):
    pipeline = local_pipeline.LocalPipeline()
    annotators: List[Annotator] = []
    langs = ["es", "zh", "fr", "it", "de"]
    model_paths = ["data/saved_models/joint/es.joint.wtype.model",
                   "data/saved_models/joint/zh.joint.wtype.model",
                   "data/saved_models/joint/fr.joint.31.5k_v2.model",
                   "data/saved_models/joint/it.joint.56.5k.10M.model",
                   "data/saved_models/joint/de.joint.20M.99k.w0.4.c0.6.model"]
    VOCABPKL = "data/{}wiki/vocab/{}wiki.train.vocab.wiki.en-{}.{}.vec_wiki.en.vec.True.True.5.0.word2idx.pkl"
    VECPKL = "data/{}wiki/vocab/{}wiki.train.vocab.wiki.en-{}.{}.vec_wiki.en.vec.True.True.5.0.embeddings.pkl"
    COHPATH = "data/{}wiki/combined_coh/en{}.coh1M"
    for lang, model_path in zip(langs, model_paths):
        vocab_pkl = VOCABPKL.format(lang, lang, lang, lang)
        vec_pkl = VECPKL.format(lang, lang, lang, lang)
        coh_path = COHPATH.format(lang, lang)
        args["lang"] = lang
        args["vocabpkl"] = vocab_pkl
        args["vecpkl"] = vec_pkl
        args["cohstr"] = coh_path
        args["restore"] = model_path
        args["filter_sizes"] = "5"
        annotator = setup_annotator(args=args, pipeline=pipeline)
        # print(model.lang)
        annotator.load_params()
        annotators.append(annotator)

    # The model should have two methods
    # 1) method load_params() that loads the relevant model parameters into memory.
    # 2) method inference_on_ta(docta, new_view_name) that takes a text annotation and view name,
    # creates the view in the text annotation, and returns it.
    # See the DummyModel class for a minimal example.
    # wrapper = MultiModelWrapperServerLocal(models=models)
    multi_annotator = MultiAnnotator(annotators=annotators)
    app.add_url_rule(rule='/annotate', endpoint='annotate', view_func=multi_annotator.annotate, methods=['GET'])
    app.run(host='0.0.0.0', port=8009)
    # On running this main(), you should be able to visit the following URL and see a json text annotation returned
    # http://127.0.0.1:5000/annotate?text="Stephen Mayhew is a person's name"&views=DUMMYVIEW

    # Chinese Example
    # 奥巴马出生於美国夏威夷州檀香山，他在夏威夷长大，但童年时期也在华盛顿州和印度尼西亚分别生活了。从哥伦比亚大学毕业之后，他在芝加哥做。奥巴马进入了哈佛法学院，在那成为了哈佛法律评论的第名非裔总编辑。毕业后他成为了名民权律师，在芝加哥大学法学院任宪制性法律教授。当选伊利诺州参议员，并担任职务直至参选联邦参议员。同年因意想不到的参议员初选胜利，在美国民主党全国代表大会上发表主题演讲和以绝对优势胜出参议员选举，成为全美知名的政治人物。

    #

if __name__ == "__main__":
    args = PARSER.parse_args()
    args = vars(args)
    main(args)
