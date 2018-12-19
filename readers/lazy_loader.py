import os
import time
import logging
import sys
from utils.mention_utils import make_mentions_from_file
import random

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)

__author__ = 'Shyam'


class LazyLoader(object):
    def __init__(self, path, idx_version=True, repeat=0, shuffle=False):
        self.reset()
        if os.path.isdir(path):
            self.files = sorted([os.path.join(path, filename) for filename in os.listdir(path)])
            # self.files = [os.path.join(path,wiki_AA)] TODO combine all training files for faster loading
        else:
            self.files = [path]
        self.num_files = len(self.files)
        self.repeat = repeat
        self.idx_version = idx_version
        self.shuffle = shuffle
        self.load_mentions_from_next_file()

    def _load_mentions_from_file(self):
        file = self.files[self.fidx]
        # logging.info("file loaded : %s", file)
        self.fidx += 1
        self.fmens = make_mentions_from_file(file,self.idx_version)

    def load_mentions_from_next_file(self):
        stime = time.time()
        self._load_mentions_from_file()
        self.num_fmens = len(self.fmens)
        self.findices = list(range(self.num_fmens))
        if self.shuffle:
            random.shuffle(self.findices)
        self.fmen_idx = 0
        ttime = (time.time() - stime) / 60.0
        # logging.info("#mentions: %d. Time : %.2f mins", self.num_fmens, ttime)

    def __iter__(self):
        return self

    def __next__(self):
        m = self.next()
        if m is None:
            raise StopIteration

    def next(self):
        if self.fmen_idx == self.num_fmens:
            if self.fidx == self.num_files:
                self.epochs += 1
                if self.epochs > self.repeat:
                    return None
                self.fidx = 0
            self.load_mentions_from_next_file()
        index = self.findices[self.fmen_idx]
        mention = self.fmens[index]
        self.fmen_idx += 1
        return mention

    def reset(self):
        self.epochs = 0
        self.fmen_idx = 0
        self.fidx = 0
        self.findices = []
        self.num_fmens = 0
        self.fmens = []


if __name__ == '__main__':
    loader = LazyLoader(sys.argv[1],repeat=1)
    seen = 0
    for idx, m in enumerate(loader):
        print(idx)
    # sys.exit(0)
    loader.reset()
    while True:
        mention = loader.next()
        if mention is None:
            break
        print(mention.surface,mention.wid)
        seen += 1
    print("total seen", seen)
    print("epoch",loader.epochs)
    loader.reset()

    while True:
        mention = loader.next()
        if mention is None:
            break
        print(mention.surface,mention.wid)
        seen += 1
    print("total seen", seen)
    print("epoch",loader.epochs)
