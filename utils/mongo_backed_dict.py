import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pymongo
from pymongo import MongoClient


class MongoBackedDict:
    def __init__(self, dbname, hostname="localhost", port=27017, insert_freq=10000):
        self.client = MongoClient(hostname, port)
        self.db = self.client['mymongo']
        self.cll = self.db[dbname]
        self.insert_freq = insert_freq
        # self.cll.create_index([("key", pymongo.HASHED)])

    def __setitem__(self, key, value):
        """
        Never use this, compute a regular map and then do a bulk_insert
        :param key:
        :param value:
        :return:
        """
        # key = self.encode(key)
        doc = {"key": key, "value": value}
        # print("doc",doc)
        # BEWARE! this does not replace, but adds another duplicate entry!
        self.cll.insert_one(doc)

    def bulk_insert(self, regular_map, value_func=None, insert_freq=10000):
        """
        Writes a regular python dict (map) to mongo.
        Common idiom is bulk_insert(regular_map=mydict,insert_freq=len(mydict))
        :param regular_map:
        :param value_func:
        :param insert_freq:
        :return:
        """
        docs = []
        for idx, k in enumerate(regular_map):
            if value_func is None:
                val = regular_map[k]
            else:
                # print("applying valfunc")
                val = value_func(regular_map[k])
            # print(val)
            docs.append({"key": k, "value": val})
            if idx > 0 and idx % insert_freq == 0:
                logging.info("inserting %d", idx)
                self.cll.insert_many(docs)
                logging.info("inserted %d", idx)
                docs = []
        # insert remaining ...
        if len(docs) > 0:
            self.cll.insert_many(docs)
        logging.info("mongo map size %d", self.size())
        logging.info("building hashed index on \"key\"")
        self.cll.create_index([("key", pymongo.HASHED)])

    def __getitem__(self, key):
        # key = self.encode(key)
        doc = self.cll.find_one({'key': key})
        if doc is None:
            return None
        val = doc["value"]
        if isinstance(val, list) and (isinstance(val[0], tuple) or isinstance(val[0], list)):
            # this should be a list of tuples
            return dict(val)
        else:
            # TODO: this should check if its dict
            return val

    def __contains__(self, key):
        doc = self.__getitem__(key)
        if doc is None:
            return False
        return True

    def encode(self, key):
        return key.replace("\\", "\\\\").replace("\$", "\\u0024").replace(".", "\\u002e")

    def size(self):
        return self.cll.count()

    def drop_collection(self):
        self.cll.drop()

    def all_iterator(self):
        for post in self.cll.find():
            yield post["key"]

    def __iter__(self):
        for post in self.cll.find():
            yield post["key"]


if __name__ == "__main__":
    dd = MongoBackedDict(dbname="mydict")
    for key in range(100):
        dd[key] = key * key
    print(dd.size())
    print(dd[11])
    for key in dd:
        # print(key)
        print(key, dd[key])
