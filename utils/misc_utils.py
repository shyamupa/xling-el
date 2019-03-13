from collections import namedtuple, defaultdict
import pickle
import sys
import logging
import os
from utils.mongo_backed_dict import MongoBackedDict
import utils.constants as K
import json
from pymongo.errors import DocumentTooLarge

logging.basicConfig(format='%(asctime)s: %(filename)s:%(lineno)d: %(message)s', level=logging.INFO)

import time

__author__ = 'Shyam'


def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def save_fast(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_json(fname, obj):
    with open(fname, 'w') as f:
        json.dump(obj, f)


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


# TODO this is useless now, use mongo
def load_langlinks(lang):
    fr2entitles, en2frtitles = load_map("data/" + lang + "wiki/idmap/fr2entitles")
    return fr2entitles, en2frtitles


def load_langlinks_mongo(lang, overwrite=False):
    fr2entitles, en2frtitles = load_map_mongo("data/" + lang + "wiki/idmap/fr2entitles", overwrite=overwrite)
    return fr2entitles, en2frtitles


# TODO this is useless now, use mongo
def load_map(path):
    pkl_path = path + ".pkl"
    if os.path.exists(pkl_path):
        logging.info("pkl found! loading map %s", pkl_path)
        m, rev_m = load(pkl_path)
    else:
        f = open(path)
        m = {}
        err = 0
        logging.info("pkl not found ...")
        logging.info("loading map from %s", path)
        for idx, l in enumerate(f):
            parts = l.strip().split("\t")
            if len(parts) != 2:
                logging.info("error on line %d %s", idx, parts)
                err += 1
                continue
            k, v = parts
            if k in m:
                logging.info("duplicate key %s was this on purpose?", k)
            m[k] = v
        rev_m = {v: k for k, v in m.items()}
        logging.info("map of size %d loaded %d err lines", len(m), err)
        logging.info("saving pkl... %s", pkl_path)
        obj = m, rev_m
        save(pkl_path, obj)
    return m, rev_m


def load_map_mongo(path, overwrite=False):
    m = MongoBackedDict(dbname=path)
    rev_m = None
    if m.size() == 0 or overwrite:
        logging.info("dropping existing collection ...")
        m.drop_collection()
        tmp = {}
        # logging.info("pkl not found ...")
        logging.info("loading map from %s", path)
        f = open(path)
        err = 0
        for idx, l in enumerate(f):
            parts = l.strip().split("\t")
            if len(parts) != 2:
                logging.info("error on line %d %s", idx, parts)
                err += 1
                continue
            k, v = parts
            if k in tmp:
                logging.info("duplicate key %s was this on purpose?", k)
            tmp[k] = v
        rev_m = {v: k for k, v in tmp.items()}
        logging.info("inserting map of size %d to mongo (%d err lines)", len(tmp), err)
        m.bulk_insert(regular_map=tmp, insert_freq=len(tmp))
    return m, rev_m


# TODO this is useless now, use mongo
def load_id2title(f):
    pkl_path = f + ".pkl"
    if os.path.exists(pkl_path):
        logging.info("found id2t pkl %s", pkl_path)
        id2t, t2id, redirect_set = load(pkl_path)
    else:
        id2t, t2id = {}, {}
        redirect_set = set([])
        for line in open(f):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                logging.info("bad line %s", line)
                continue
            # page_id, title = parts
            page_id, page_title, is_redirect = parts
            id2t[page_id] = page_title
            t2id[page_title] = page_id
            if is_redirect == "1":
                redirect_set.add(page_title)
        obj = id2t, t2id, redirect_set
        save(pkl_path, obj)
        logging.info("saving id2t pkl to %s", pkl_path)
    logging.info("id2t of size %d", len(id2t))
    return id2t, t2id, redirect_set


def load_id2title_mongo(path, overwrite=False):
    mongo_id2t = MongoBackedDict(dbname=path + ".id2t")
    # TODO Maybe you can use the same db and its reverse?
    mongo_t2id = MongoBackedDict(dbname=path + ".t2id")
    # TODO fix below
    redirect_set = None
    if mongo_id2t.size() == 0 or mongo_t2id.size() == 0 or overwrite:
        logging.info("db not found at %s. creating ...", path)
        id2t, t2id = {}, {}
        redirect_set = set([])
        for line in open(path):
            parts = line.strip().split("\t")
            if len(parts) != 3:
                logging.info("bad line %s", line)
                continue
            # page_id, title = parts
            page_id, page_title, is_redirect = parts
            id2t[page_id] = page_title
            t2id[page_title] = page_id
            if is_redirect == "1":
                redirect_set.add(page_title)
        mongo_id2t.bulk_insert(regular_map=id2t, insert_freq=len(id2t))
        mongo_t2id.bulk_insert(regular_map=t2id, insert_freq=len(t2id))
        # obj = id2t, t2id, redirect_set
        # save(pkl_path, obj)
        # logging.info("saving id2t pkl to %s", pkl_path)
    logging.info("id2t of size %d", mongo_id2t.size())
    logging.info("t2id of size %d", mongo_t2id.size())
    return mongo_id2t, mongo_t2id, redirect_set


def load_disamb2title(f):
    id2t, t2id = load_map(f)
    return id2t, t2id


# TODO this is useless now, use mongo
def load_redirects(path):
    pkl_path = path + ".pkl"
    if os.path.exists(pkl_path):
        logging.info("pkl found! loading map %s", pkl_path)
        redirect2title = load(pkl_path)
    else:
        f = open(path)
        redirect2title = {}
        err = 0
        logging.info("pkl not found ...")
        logging.info("loading map from %s", path)
        for idx, l in enumerate(f):
            parts = l.strip().split("\t")
            if len(parts) != 2:
                logging.info("error on line %d %s", idx, parts)
                err += 1
                continue
            redirect, title = parts
            if redirect in redirect2title:
                logging.info("duplicate keys! was this on purpose?")
            redirect2title[redirect] = title
        logging.info("map of size %d loaded %d err lines", len(redirect2title), err)
        logging.info("saving pkl... %s", pkl_path)
        obj = redirect2title
        save(pkl_path, obj)
    logging.info("r2t of size %d", len(redirect2title))
    return redirect2title


def load_redirects_mongo(path, overwrite=False):
    # pkl_path = path + ".pkl"
    # if os.path.exists(pkl_path):
    # logging.info("pkl found! loading map %s", pkl_path)
    # r2t = load(pkl_path)
    # else:
    mongo_r2t = MongoBackedDict(dbname=path)
    if mongo_r2t.size() == 0 or overwrite:
        logging.info("db not found at %s. creating ...", path)
        f = open(path)
        r2t = {}
        err = 0
        logging.info("pkl not found ...")
        logging.info("loading map from %s", path)
        for idx, l in enumerate(f):
            parts = l.strip().split("\t")
            if len(parts) != 2:
                logging.info("error on line %d %s", idx, parts)
                err += 1
                continue
            redirect, title = parts
            if redirect in r2t:
                logging.info("duplicate keys! was this on purpose?")
            r2t[redirect] = title
        logging.info("map of size %d loaded %d err lines", len(r2t), err)
        mongo_r2t.bulk_insert(regular_map=r2t, insert_freq=len(r2t))
    logging.info("r2t of size %d", mongo_r2t.size())
    return mongo_r2t


NamedEntity = namedtuple('NamedEntity', ['wid', 'title', 'mid', 'types', 'count'])


def load_nekb(kbfile):
    # ="data/enwiki/wid_title_mid_types_counts.txt"
    pkl_path = kbfile + ".nekb.pkl"
    if os.path.exists(pkl_path):
        logging.info("pkl found! loading map %s", pkl_path)
        wid2ne, mid2ne, title2ne = load(pkl_path)
    else:
        logging.info("pkl not found! making nekb maps...")
        wid2ne, mid2ne, title2ne = {}, {}, {}
        for idx, line in enumerate(open(kbfile)):
            parts = line.strip().split("\t")
            # print(parts)
            title, wid, mid, types, cnt = parts
            cnt = int(cnt)
            types = types.split(" ")
            ne = NamedEntity(wid, title, mid, types, cnt)
            wid2ne[ne.wid] = ne
            mid2ne[ne.mid] = ne
            title2ne[ne.title] = ne
        obj = wid2ne, mid2ne, title2ne
        save(pkl_path, obj)
    return wid2ne, mid2ne, title2ne


# TODO this is useless now, use mongo
def load_prob_map(out_prefix, kind):
    path = out_prefix + "." + kind
    pkl_path = path + ".pkl"
    if os.path.exists(pkl_path):
        logging.info("pkl found! %s", pkl_path)
        mmap = load(pkl_path)
    else:
        mmap = defaultdict(lambda: defaultdict(float))
        logging.info("loading from %s", path)
        for idx, line in enumerate(open(path)):
            parts = line.split("\t")
            if len(parts) != 4:
                logging.info("error on line %d: %s", idx, line)
                continue
            y, x, prob, _ = parts
            mmap[y][x] = float(prob)
        logging.info("pkling ... %s", pkl_path)

        pkl_map = {}
        for y in mmap:
            if y not in pkl_map: pkl_map[y] = {}
            for x in mmap[y]:
                pkl_map[y][x] = mmap[y][x]
        save(pkl_path, pkl_map)
    return mmap


def load_prob_map_mongo(out_prefix, kind, dbname=None, force_rewrite=False):
    path = out_prefix + "." + kind
    if dbname is None:
        dbname = path
    logging.info("dbname is %s", dbname)
    probmap = MongoBackedDict(dbname=dbname)
    logging.info("reading collection %s", path)
    if probmap.size() > 0 and not force_rewrite:
        logging.info("collection already exists in db (size=%d). returning ...", probmap.size())
        return probmap
    else:
        if force_rewrite:
            logging.info("dropping existing collection in db.")
            probmap.drop_collection()
        # mmap = defaultdict(lambda: defaultdict(float))
        mmap = {}
        for idx, line in enumerate(open(path)):
            parts = line.split("\t")
            if idx > 0 and idx % 1000000 == 0:
                logging.info("read line %d", idx)
            if len(parts) != 4:
                logging.info("error on line %d: %s", idx, line)
                continue
            y, x, prob, _ = parts
            if y not in mmap:
                mmap[y] = {}
            mmap[y][x] = float(prob)
        for y in list(mmap.keys()):
            # TODO will below ever be false?
            # if y not in probmap:
            # Nested dict keys cannot have '.' and '$' in mongodb
            # tmpdict = {x: mmap[y][x] for x in mmap[y]}
            if len(mmap[y]) > 5000:
                logging.info("string %s can link to %d items (>10k)... skipping", y, len(mmap[y]))
                # mmap[y] = []
                # continue
                del mmap[y]
            else:
                # tmpdict = [(x, mmap[y][x]) for x in mmap[y]]
                mmap[y] = list(mmap[y].items())
                # try:
                #     probmap[y] = tmpdict
                # except DocumentTooLarge as e:
                #     print(y, len(tmpdict))
                #     print(e)
        probmap.bulk_insert(regular_map=mmap, insert_freq=len(mmap))
    return probmap


def load_vocab(path, wid=False):
    pkl_path = path + ".pkl"
    if os.path.exists(pkl_path):
        logging.info("pkl found! loading map %s", pkl_path)
        m, rev_m = load(pkl_path)
    else:
        f = open(path)
        if wid:
            m = {K.NULL_TITLE_WID: K.NULL_TITLE_ID}
        else:
            m = {K.OOV_TOKEN: K.OOV_ID}
        logging.info("loading vocab from %s", path)
        for idx, l in enumerate(f):
            word = l.strip()
            m[word] = idx + 1
        rev_m = {v: k for k, v in m.items()}
        logging.info("vocab of size %d loaded", len(m))
        logging.info("saving pkl... %s", pkl_path)
        obj = m, rev_m
        save(pkl_path, obj)
    return m, rev_m


def read_candidates_dict(path):
    ddict = {}
    f = open(path)
    missing_gold = 0
    for linum, line in enumerate(f):
        parts = line.strip().split("\t")
        surface, gold_wid, was_missed, candidates = parts[0], parts[1], parts[2], parts[3:]
        surface = surface[len("surface:"):]
        gold_wid = gold_wid[len("gold_wid:"):]
        candidates = [c.split('|') for c in candidates]
        cand_w_labels = []
        ###############
        ans = [(gold_wid, 0.0, 1)]
        found = False
        for c in candidates:
            title, wid, p_t_given_s, label = c[0], c[1], float(c[2]), int(c[4])
            # CAREFUL wid is STRING
            if gold_wid == wid:
                found = True
                ans[0] = (gold_wid, p_t_given_s, 1)
            else:
                ans.append((wid, p_t_given_s, label))
        if not found and len(candidates) > 0:
            ans = ans[:len(candidates)]
        wids, wid_cprobs, isgolds = zip(*ans)
        wids, wid_cprobs, isgolds = list(wids), list(wid_cprobs), list(isgolds)
        ddict[(surface, gold_wid)] = wids, wid_cprobs, isgolds
        ###############
        #######REPLACE########
        # for c in candidates:
        #     title, wid, p_t_given_s, label = c[0], c[1], float(c[2]), int(c[4])
        #     # CAREFUL wid is STRING
        #     cand_w_labels.append((title, wid, p_t_given_s, label))
        # found = False
        # for c_w_l in cand_w_labels:
        #     if c_w_l[-1] == 1:
        #         if found:
        #             print("cannot have more than one gold in candidates!")
        #             sys.exit(0)
        #         found = True
        # if not found:
        #     logging.info("missing gold %s %s", surface, gold_wid)
        #     print(surface, gold_wid, cand_w_labels)
        #     missing_gold += 1
        #     continue
        # cand_w_labels = sorted(cand_w_labels, key=lambda cand: -1 * cand[-1])
        # _, wids, wid_cprobs, isgolds = zip(*cand_w_labels)
        # wids, wid_cprobs, isgolds = list(wids), list(wid_cprobs), list(isgolds)
        # ddict[(surface, gold_wid)] = wids, wid_cprobs, isgolds
        #######REPLACE########
    logging.info("#%d missed gold in candidates!", missing_gold)
    return ddict