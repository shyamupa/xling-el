#!/usr/bin/env bash
ME=`basename $0` # for usage message

if [[ "$#" -ne 3 ]]; then 	# number of args
    echo "USAGE: $ME <lang> <infile> <outfile>"
    exit
fi
lang=$1
infile=$2
outfile=$3
VOCABPKL="data/${lang}wiki/vocab/${lang}wiki.train.vocab.wiki.en-${lang}.${lang}.vec_wiki.en.vec.True.True.5.0.word2idx.pkl"
VECPKL="data/${lang}wiki/vocab/${lang}wiki.train.vocab.wiki.en-${lang}.${lang}.vec_wiki.en.vec.True.True.5.0.embeddings.pkl"
COHPATH="data/${lang}wiki/combined_coh/en${lang}.coh1M"

case ${lang} in
    de)
        restore_path=data/saved_models/joint/de.joint.20M.99k.w0.4.c0.6.model
        ;;
    es)
        restore_path=data/saved_models/joint/es.joint.wtype.model
        ;;
    fr)
        restore_path=data/saved_models/joint/fr.joint.31.5k_v2.model
        ;;
    it)
        restore_path=data/saved_models/joint/it.joint.56.5k.10M.model
        ;;
    zh)
        restore_path=data/saved_models/joint/zh.joint.wtype.model
        ;;
esac

python -m readers.xel_annotator \
       --kb_file data/mykbs/biggest.kb \
       --vocabpkl ${VOCABPKL} \
       --vecpkl ${VECPKL} \
       --ncands 20 \
       --usecoh \
       --cohstr ${COHPATH} \
       --test_doc ${infile} \
       --out_doc ${outfile} \
       --restore ${restore_path} \
       --lang ${lang}



if [[ $? == 0 ]]        # success
then
    :                   # do nothing
else                    # something went wrong
    echo "SOME PROBLEM OCCURED";            # echo file with problems
fi
