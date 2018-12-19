import sys
import utils.constants as K
import logging

__author__ = 'Nitish,Shyam'


class Mention(object):

    def __init__(self, mention_line, idx_version=False):
        ''' mention_line : Is the string line stored for each mention
        mid wid wikititle start_token end_token surface tokenized_sentence all_types
        '''
        mention_line = mention_line.strip()
        self.mention_line = mention_line
        parts = mention_line.split("\t")
        self.idx_version = idx_version
        if len(parts) < 3:
            logging.warning("bad line %s",mention_line)
            sys.exit(0)
        self.mid, self.wid, self.wikititle = parts[0:3]
        self.start_token = int(parts[3]) + 1  # Adding <s> in the start
        self.end_token = int(parts[4]) + 1
        self.surface = parts[5]
        self.sent_tokens = [K.START_ID] if idx_version else [K.START_WORD]
        tokens = list(map(int, parts[6].split(" "))) if idx_version else parts[6].split(" ")
        self.sent_tokens.extend(tokens)
        end = K.END_ID if idx_version else K.END_WORD
        self.sent_tokens.append(end)
        self.types = list(map(int, parts[7].split(" "))) if idx_version else parts[7].split(" ")
        self.coherence = [K.OOV_TOKEN]
        if len(parts) > 8:  # If no mention surface words in coherence
            if parts[8].strip() == "":
                self.coherence = [K.OOV_ID] if idx_version else [K.OOV_TOKEN] #[unk_word]
            else:
                self.coherence = list(map(int, parts[8].split(" "))) if idx_version else parts[8].split(" ")
            # if parts[8].strip() == "":
            #     self.coherence = [K.OOV_TOKEN]  # [unk_word]
            # else:
            #     self.coherence = parts[8].split(" ")
        # if len(parts) == 10:
        #     self.docid = parts[9]
        if len(parts) == 11 and parts[9]!='null':
            # print(parts[9])
            self.doc_bow = parts[9].split(' ')
            self.doc_bow = [s.split(":=") for s in self.doc_bow]
            self.doc_bow = [(w, int(c)) for w, c in self.doc_bow]
            # print(len(self.doc_bow))
        if self.end_token > (len(self.sent_tokens) - 1):
            # logging.info("Bad Line #: %s", mention_line)
            logging.info("Bad Line")

    def __str__(self):
        outstr = self.wid + "\t"
        outstr += self.wikititle + "\t"
        for i in range(len(self.sent_tokens)):
            if i == self.start_token and i == self.end_token:
                fmt = "<<%d>> " % self.sent_tokens[i] if self.idx_version else "<<%s>> " % self.sent_tokens[i]
                outstr += fmt
                continue
            if i == self.start_token:
                fmt = "<<%d " % self.sent_tokens[i] if self.idx_version else "<<%s " % self.sent_tokens[i]
                outstr += fmt
                continue
            if i == self.end_token:
                fmt = "%d>> " % self.sent_tokens[i] if self.idx_version else "%s>> " % self.sent_tokens[i]
                outstr += fmt
                continue
            fmt = "%d " % self.sent_tokens[i] if self.idx_version else "%s " % self.sent_tokens[i]
            outstr += fmt

        outstr = outstr.strip()
        return outstr


if __name__ == '__main__':
    for line in open(sys.argv[1]):
        m = Mention(line)
        print(m)
