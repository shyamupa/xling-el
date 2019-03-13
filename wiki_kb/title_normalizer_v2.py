import sys
import logging
from utils.misc_utils import load_id2title_mongo, load_redirects_mongo
import utils.constants as K

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)
__author__ = 'Shyam'


class TitleNormalizer:
    def __init__(self, lang="en", redirect_map=None, t2id=None, id2t=None, redirect_set=None,date='20170520', prep_lower2upper=False):
        if t2id is None:
            id2t, t2id, redirect_set = load_id2title_mongo('data/{}wiki/idmap/{}wiki-{}.id2t'.format(lang,lang,date))
        if redirect_map is None:
            redirect_map = load_redirects_mongo('data/{}wiki/idmap/{}wiki-{}.r2t'.format(lang,lang,date))
        self.null_counts = 0
        self.call_counts = 0
        self.lang = lang
        self.redirect_map = redirect_map
        self.title2id, self.id2title, self.redirect_set = t2id, id2t, redirect_set
        # for wid in self.id2title:
        #     print(wid, type(wid), self.id2title[wid])
        if prep_lower2upper:
            self.lower2upper = {title.lower():title for title in self.title2id}
            for redirect in self.redirect_map:
                self.lower2upper[redirect.lower()] = self.redirect_map[redirect]

    def get_id2title(self, wid: str):
        wiki_title = self.id2title[wid]
        if wiki_title is None:
            return K.NULL_TITLE
        wiki_nrm = self.normalize(wiki_title)
        return wiki_nrm

    def normalize(self, title):
        """

        """
        # TODO disambiguation pages should ideally go to NULLTITLE
        self.call_counts += 1
        # Check this first, because now tid can contains tids for titles that are redirect pages.
        if title in self.redirect_map:
            return self.redirect_map[title]

        if title in self.title2id:
            return title

        title_tokens = title.split('_')
        title = "_".join([t.capitalize() for t in title_tokens])

        if title in self.redirect_map:
            return self.redirect_map[title]

        if title in self.title2id:
            return title

        self.null_counts += 1
        return K.NULL_TITLE

    # def __del__(self):
    #     logging.info("dying ... title nrm saw %d/%d nulls/calls", self.null_counts, self.call_counts)


"""
TODO
test for normalization
anna_Kurnikova
nasa
2_Ronnies
ActresseS
AN.O.VA.
annova
cyanide
"""

if __name__ == '__main__':
    title_normalizer = TitleNormalizer(lang=sys.argv[1])
    try:
        while True:
            surface = input("enter title:")
            nrm = title_normalizer.normalize(surface)
            logging.info("normalized title %s",nrm)
            # wid = input("enter title:")
            # nrm = title_normalizer.get_id2title(wid=wid)
            # logging.info("normalized title %s",nrm)
    except KeyboardInterrupt:
        print('interrupted!')
