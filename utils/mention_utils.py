import time

from datastructs.mention import Mention


def make_mentions_from_file(mens_file, idx_version=True, verbose=False):
    stime = time.time()
    with open(mens_file, 'r') as f:
        mention_lines = f.read().strip().split("\n")
        mentions = []
        for line in mention_lines:
            mentions.append(Mention(line,idx_version=idx_version))
            ttime = (time.time() - stime)
    if verbose:
        filename = mens_file.split("/")[-1]
        print(" ## Time in loading {} mens : {} secs".format(mens_file, ttime))
    return mentions




