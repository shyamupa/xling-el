import logging

import numpy as np

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)
from collections import defaultdict
from wiki_kb.candidate import Candidate

__author__ = 'Shyam'


def combine_duplicates_n_sort(cands):
    # TODO This needs to be more precise.
    seen = defaultdict(list)
    for cand in cands: seen[cand.en_title].append(cand)
    unique_cands = []
    for en_title in seen:
        title_cands = seen[en_title]
        if len(title_cands) > 1:
            rep = title_cands[0]  # representative
            lang, surface, fr_title, edit = rep.lang, rep.surface, rep.fr_title, rep.inv_edit_dist
            # Keep the max probability
            # combined_p_s_given_t = np.max([cand.p_s_given_t for cand in title_cands])
            combined_p_t_given_s = np.max([cand.p_t_given_s for cand in title_cands])
            new_cand = Candidate(surface=surface, en_title=en_title, fr_title=fr_title, is_gold=0,
                                 p_t_given_s=combined_p_t_given_s,
                                 lang=lang, src="combined")
            unique_cands.append(new_cand)
        else:
            unique_cands.append(title_cands[0])
    unique_cands = sorted(unique_cands, key=lambda cand: -1.0 * cand.p_t_given_s)
    return unique_cands