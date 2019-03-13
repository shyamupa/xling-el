Code for running the entity linking model. This is part of the code
for the [xelms](https://github.com/shyamupa/xelms) project.


#### Requirements

1. pytorch (0.2.0+21f8ad4): installed from source, and patched for sparse tensor operations (instructions below).
2. python3.
3. [cogcomp-nlpy](https://github.com/CogComp/cogcomp-nlpy/). 
4. Download the resources and trained models [here](http://bilbo.cs.illinois.edu/~upadhya3/data_release_v2.tar.gz) and place them in the folder `xling-el/data`. Right now, pre-trained models are available for German, Spanish, French, Italian, and Chinese.


#### Resources for Candidate Generation

1. First set up candidate generation and other resources as described in
projects
[wikidump_preprocessing](https://github.com/shyamupa/wikidump_preprocessing)
and [wiki_candgen](https://github.com/shyamupa/wiki_candgen). 
2. A mongo server instance is needed that uses the databases constructed in [wiki_candgen](https://github.com/shyamupa/wiki_candgen).
   

#### Patching Pytorch for Sparse Tensor Operations

This is best done in a new conda environment.

1. First checkout the `sparse_patch` branch from [this](https://github.com/shyamupa/pytorch) repository.
```bash
git clone https://github.com/shyamupa/pytorch
cd pytorch
git checkout sparse_patch
```
2. Install the patched code from source using the following commands,  

```bash
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]

# Install basic dependencies
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c mingfeima mkldnn
cd pytorch_patched
python setup.py install
```

Ensure that the patched pytorch was successfully installed,

```python
>>> import torch
>>> torch.__version__
'0.2.0+43662e7'
```

### Mention Detection using NER
1. For German, Spanish, French and Italian, download relevant [Spacy](https://spacy.io/models/) NER Models 
```bash
pip install spacy
python -m spacy download de_core_news_sm
python -m spacy download es_core_news_md
python -m spacy download fr_core_news_md
python -m spacy download it_core_news_sm
```

2. For Chinese, download [stanford corenlp jar](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) and the [chinese model jar](http://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-10-05-models.jar) and place them in a `stanford_jars` directory.
```
$ ls stanford_jars/
stanford-corenlp-full-2018-10-05
$ ls stanford_jars/stanford-corenlp-full-2018-10-05
...
...
stanford-chinese-corenlp-2018-10-05-models.jar
...
```
And set the bash environment variable `CORENLP_HOME` to `path/to/stanford_jars/stanford-corenlp-full-2018-10-05`.
```bash
export CORENLP_HOME=path/to/stanford_jars/stanford-corenlp-full-2018-10-05
```

## Running the Model
To run the model, use the command,
```bash
./run_inference_on_doc.sh <lang> <infile> <outfile>
```

For instance, for running on a German document `test_docs/de_doc.txt`, one would run

```
./run_inference_on_doc.sh de test_docs/de_doc.txt test_docs/de_doc_output.txt
```

The json output will be produced in `test_docs/de_doc_output.txt`. 

## Output

The output file is a json serialized text annotation, with a view named `NEURAL_XEL_<lang>`. The view consists of a list of the 
constituents that have been linked to a Wikipedia title. Below is the output for the German test document provided in the repo,

```json
...
"viewName": "NEURAL_XEL_de",
...
...
"constituents": [
      {
       "end": 2,
       "label": "en.wikipedia.org/wiki/Angela_Merkel",
       "score": 0.5128146075318596,
       "start": 0,
       "tokens": "Angela Merkel"
      },
      {
       "end": 5,
       "label": "NULLTITLE",
       "score": 0.05000000074505806,
       "start": 4,
       "tokens": "Elim-Krankenhaus"
      },
      ...
```

The label field for each constituent is the predicted Wikipedia entity for the span identified by the `start` and `end` token index. 
Here a label of `NULLTITLE` means that the named entity detected by the mention detection system could not be linked to any entity. 