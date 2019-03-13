from ccg_nlpy import local_pipeline
from flask import Flask
from flask_cors import CORS

from readers.xel_annotator import setup_annotator
from utils.arguments import PARSER

app = Flask(__name__)
# necessary for testing on localhost
CORS(app)


def main(args):
    pipeline = local_pipeline.LocalPipeline()

    annotator = setup_annotator(args=args, pipeline=pipeline)
    annotator.load_params()
    # The model should have two methods
    # 1) method load_params() that loads the relevant model parameters into memory.
    # 2) method inference_on_ta(docta, new_view_name) that takes a text annotation and view name,
    # creates the view in the text annotation, and returns it.
    # See the DummyModel class for a minimal example.
    app.add_url_rule(rule='/annotate', endpoint='annotate', view_func=annotator.annotate, methods=['GET'])
    # app.run(host='0.0.0.0', port=8009)
    app.run(host='0.0.0.0', port=8080)
    # On running this main(), you should be able to visit the following URL and see a json text annotation returned
    # http://127.0.0.1:5000/annotate?text="Stephen Mayhew is a person's name"&views=DUMMYVIEW


if __name__ == "__main__":
    args = PARSER.parse_args()
    args = vars(args)
    main(args)
