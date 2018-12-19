import logging

from readers.lazy_loader import LazyLoader


def pad_batch(batch, pad_unit):
    # keras_utils is slower
    lengths = [len(i) for i in batch]
    max_length = max(lengths)
    for ex in batch:
        padding = (max_length - len(ex)) * [pad_unit]
        ex += padding
    return batch, lengths


# def sort_by_length(padded_batch, lengths):
#     srt = sorted(zip(lengths, padded_batch), key=lambda pair: -pair[0])
#     sorted_padded_batch = [padded_sent for _, padded_sent in srt]
#     sorted_lengths = [length for length, _ in srt]
#     return sorted_padded_batch, sorted_lengths


class MentionReader(object):
    def __init__(self):
        self.loader = None
        self.finished = False

    def __iter__(self):  # needed to make iterator
        return self

    def __next__(self):
        return self.next_batch()

    def next_raw_batch(self):
        return self._next_batch()

    def next_batch(self):
        return self._next_padded_batch()

    def reset(self):
        self.finished = False
        self.loader.reset()

    def epochs_done(self):
        return self.loader.epochs

    def _read_mention(self):
        mention = self.loader.next()
        return mention

    def load_loader(self, iters, istest, fpath, shuffle):
        if istest:
            repeats = 0 # should be 0
        else:
            repeats = iters - 1
        self.loader = LazyLoader(fpath, shuffle=shuffle, repeat=repeats)

    # def load_validation(self, val_path, test_path, shuffle=False):
    #     logging.info("Validation File:%s", val_path)
    #     self.loaders["val"] =
    #     logging.info("Test File:%s", test_path)
    #     # TODO hacky way to cycle around test data
    #     self.loaders["test"] = LazyLoader(test_path,shuffle=shuffle,repeat=99999)
    #
    # def load_train(self, train_path, shuffle=True):
    #     self.loaders["train"] = LazyLoader(train_path,shuffle=shuffle, repeat=99999) # infinite repeats
    #     logging.info("Training Mention Files : %d files", self.loaders["train"].num_files)

    def _next_padded_batch(self):
        raise NotImplementedError

    def _next_batch(self):
        raise NotImplementedError
