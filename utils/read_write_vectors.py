"""Utils to read word vectors.

Also normalizes and removes accents, diacritics etc. if required
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import math
import numpy as np
import sys
import unidecode
import logging

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)

import re


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def read_word_vectors(word_vecs=None,
                      filename=None,
                      delim=' ',
                      norm=True,
                      scale=1.0,
                      lower=True,
                      remove_accents=False):
    """Read all the word vectors and normalize them.

  Also takes word_vecs in case you want to read multiple vector files into
  same word2vec map. Just keep passing it a new filename and the old word_vecs
  dictionary

  Args:
    word_vecs: if None a new dict is returned, otherwise new words are added to
      old one.
    filename: file to read vecs from. should have word2vec like format.
    delim: delimiter in the file (default ' ')
    norm: whether to normalize vectors after reading.
    remove_accents: whether to remove accents/diacritics from words for
      languages like Turkish.

  Returns:
    a dictionary of word to vectors.


  """
    # if starting afresh
    if word_vecs is None:
        word_vecs = {}

    logging.info('Reading word vectors from file:%s', filename)
    err = 0
    if filename.endswith('.gz'):
        file_object = gzip.open(filename, 'r')
    else:
        file_object = open(filename, 'r')

    for line_num, line in enumerate(file_object):
        if line_num == 0:
            parts = line.strip().split(delim)
            if len(parts)==2:
                dim = int(parts[1])
                logging.info("reading vecs of dim %d",dim)
                continue
            else:
                dim = len(parts) - 1
                logging.info("reading vecs of dim %d",dim)

        line = line.strip()

        if lower:
            line = line.lower()
        
        parts = line.split(delim)[1:]
        if len(parts) != dim:
            # logging.info("#%d error in line %d", err, line_num)
            # logging.info("line started %s", line[0:5])
            err += 1
            continue
        word = line.split(delim)[0]
        if remove_accents: word = unidecode.unidecode(word)
        if word in word_vecs:
            # logging.info("word %s already seen! skipping ...",word)
            continue
        word_vecs[word] = np.zeros(dim, dtype=float)
        for index, vec_val in enumerate(parts):
            word_vecs[word][index] = float(vec_val)

        if norm:
            word_vecs[word] /= math.sqrt((word_vecs[word] ** 2).sum() + 1e-6)
            word_vecs[word] *= scale
    logging.info("total %d lines read", line_num)
    logging.warning('%d vectors read from %s with dim %d, norm=%s accents=%s', len(word_vecs),
                    filename, dim, norm, remove_accents)
    return word_vecs


if __name__ == '__main__':
    pass
