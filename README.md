Code for running the entity linking model. This is part of the code for the [xelms](https://github.com/shyamupa/xelms) project. 

First set up candidate generation and other resources as described in projects [wikidump_preprocessing](https://github.com/shyamupa/wikidump_preprocessing) and [wiki_candgen](https://github.com/shyamupa/wiki_candgen). You need them for candidate generation. 

### Requirements

1. pytorch (0.2.0)
2. python3

### Running with detected mentions 
The current code assumes that the entity mentions have already been detected and candidates have been precomputed for the mentions.

0. Download the resources [here](http://bilbo.cs.illinois.edu/~upadhya3/data_release_v1.tar.gz) in a folder `xling-el/data`. 

1. Prepare your test mentions in a file `test_mentions_str` in the following format (each line is a single mention).

```bash
 mid <tab> wiki_id <tab> wikititle <tab> start_offset <tab> end_offset <tab> mention_surface <tab> mention_sentence <tab> types <tab> other_mention_surfaces <tab> null
```

where `start_offset` is the (0-indexed and inclusive) start offset of the entity mention in the `mention_sentence` and `end_offset` is the end offset (inclusive) of the entity mention, `mention_sentence` is atmost 50 (25 either side) tokens (lowercased) around the mention. 

You can leave `mid`, `wiki_id`, `wikititle` blank.
`other_mention_surfaces` is a list of space separated surfaces of other mentions in the same document. 

An example test file is provided in `examples/example_mentions_str`.

2. Then, generate candidates using the candidate generator in [wiki_candgen](https://github.com/shyamupa/wiki_candgen) project as follows.

```bash
  python -m wiki_kb.candidate_writer \
           --lang <LANGCODE> \
           --ncands 20 \
           --kbfile data/mykbs/example.kb \
           --mention_dir examples/test_mentions_str \
           --out cands.k20
```

An example candidate file is provided in `examples/example.cands.k20`. 

3. Convert the test mentions to ids in the vocab using the `file_convertor.py` script.

```bash
python file_convertor.py \
           --in path/to/test_mentions_str \
           --out path/to/test_mentions_ids \
           --vocabpkl path/to/vocab_pickle
```

where the vocab pickle is the file in the `data/vocabs` folder with the suffix `*.word2idx.pkl`.  

4. Run the model as follows,

```bash
python main.py \
--kb_file data/mykbs/example.kb \
--vocabpkl path/to/vocab_pickle \
--vecpkl path/to/vec_pickle \
--ncands 20 \
--ftest path/to/test_mentions_ids \
--ttcands cands.k20 \
--dump /path/to/output_file \
--restore path/to/model  
```

where the vec_pickle is the file with the suffix `*.embeddings.pkl`.
The output file is ordered the same as the input test file, with the format, 
```
mention_sentence<tab><curids and scores>
``` 
where curid is the wikipedia page id (e.g., 846720 points to the wikipedia page [Bezirk](https://en.wikipedia.org/?curid=846720) which can visited as `https://en.wikipedia.org/?curid=846720)` and scores are the model scores. The highest scoring curid is the prediction.

### Running on raw text

TODO: We will eventually support working directly with raw text. This functionality will be built using the candidate generator and the [cogcomp-nlpy](https://github.com/CogComp/cogcomp-nlpy/) package.  
