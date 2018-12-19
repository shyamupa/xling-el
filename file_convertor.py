import argparse
import os
import sys
from utils.misc_vocab_loader import VocabLoader
from utils.parallel_utils import run_as_parallel
from utils.vocab_utils import get_idx, get_typeidx
import utils.constants as K
import logging

__author__ = 'Shyam'


def process_line(line, word2idx, type2idx, coh2idx, dotype=False):
    parts = line.strip().split("\t")
    mid, wid, wikititle = parts[0:3]
    start_token = int(parts[3])  # DO NOT add <s> in the start, this is 0-indexed start token
    end_token = int(parts[4])  # this is 0-indexed end token
    surface = parts[5]
    # NO NEED TO PUT START AND END TOKEN ID
    words = parts[6].split(" ")
    # print(surface,start_token,end_token,list(enumerate(words)))
    sent_tokens = list(map(lambda word: str(get_idx(word, word2idx)), words))
    sent_str = " ".join(sent_tokens)
    if dotype:
        types = list(map(lambda t: str(get_typeidx(t, type2idx)), parts[7].split(" ")))
        types_str = " ".join(types)
    else:
        types_str = parts[7]

    if len(parts) > 8:  # If no mention surface words in coherence
        if parts[8].strip() == "":
            # coherence = [K.OOV_TOKEN]  # [unk_word]
            coherence = [str(K.OOV_ID)]
        else:
            coherence = parts[8].split(" ")
            coherence = [str(get_idx(coh,coh2idx)) for coh in coherence]
    else:
        coherence = [str(K.OOV_ID)]
    tmp = sorted(list(set(coherence)))
    coherence_str = " ".join(tmp)

    # if len(parts) == 10:
    #     docid = parts[9]
    if len(parts) == 11:
        dbow = parts[9].split(' ')
        dbow = [d.split(":=") for d in dbow]
        dbow = [[str(get_idx(d[0],word2idx)),d[1]] for d in dbow]
        dbow = [":=".join(d) for d in dbow]
        dbow_str = " ".join(dbow)
    else:
        dbow_str = " "
    buf = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (mid, wid, wikititle,
                                                    start_token, end_token,
                                                    surface, sent_str,
                                                    types_str, coherence_str, dbow_str)

    # assert end_token <= len(sent_tokens), "Line : %s" % line
    if end_token > len(sent_tokens):
        logging.info("Bad Line #: %s", line)
        # logging.info("Bad Line")
    return buf


def process_file(infile, outfile, args):
    logging.info("input %s", infile)
    logging.info("output %s", outfile)
    with open(outfile, "w") as out:
        for line in open(infile):
            buf = process_line(line,
                               word2idx=word2idx,
                               type2idx=type2idx,
                               coh2idx=coh2idx,
                               dotype=not args["notype"])
            #  mid wid wikititle start_token end_token surface tokenized_sentence all_types
            out.write(buf)


def handle_file(id, jobs_queue):
    while True:
        job = jobs_queue.get()
        if job:
            infile, outfile, args = job
            process_file(infile, outfile, args)
        else:
            logging.debug('Quit extractor')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert strings to ids')
    parser.add_argument('--in', type=str, required=True, help='input mention file with strings')
    parser.add_argument('--out', type=str, required=True, help='outfile mention file with ids')
    parser.add_argument('--vocabpkl', type=str, required=True, help='# the *.word2idx.pkl file')
    parser.add_argument('--type_vocab', type=str, default="data/enwiki/fbtypelabels.vocab", help='type vocab')
    parser.add_argument('--notype', action="store_true")
    args = parser.parse_args()
    args = vars(args)
    print(args)
    loader = VocabLoader()
    loader.load_word2idx(word2idx_pkl_path=args["vocabpkl"])
    loader.load_type_vocab(path=args["type_vocab"])
    word2idx = loader.word2idx
    coh2idx = None
    type2idx = loader.type2idx

    if os.path.isdir(args["in"]):
        wikipath = args["in"]  # "data/enwiki/enwiki_mentions_with_es-en_merged/"
        outdir = args["out"]
        
        jobs = []
        for filename in sorted(os.listdir(wikipath)):
            infile = os.path.join(wikipath, filename)
            outfile = os.path.join(outdir, filename + ".ids")
            job = (infile, outfile, args)
            jobs.append(job)  # goes to any available extract_process
        # print(jobs[:5])
        run_as_parallel(jobs_list=jobs, worker_func=handle_file)
    else:
        infile = args["in"]
        outfile = args["out"]
        job = infile, outfile
        process_file(infile, outfile, args)
