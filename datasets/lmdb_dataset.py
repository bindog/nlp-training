import os
import lmdb
import json
import pickle
import logging
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        logger.info("Initialize LMDB Datasets with path: " + lmdb_path)
        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            self.lmdb_keys = [key for key, _ in txn.cursor()]
        logger.info("LMDB Datasets Done.")

    def get_raw_dict(self, index):
        with self.env.begin(write=False) as txn:
            key = self.lmdb_keys[index]
            buf = txn.get(key)
            raw_dict = pickle.loads(buf)
            return raw_dict

    def process(self, json_str):
        raise NotImplementedError

    def __getitem__(self, index):
        raw_dict = self.get_raw_dict(index)
        return self.process(raw_dict)

    def __len__(self):
        return self.length
