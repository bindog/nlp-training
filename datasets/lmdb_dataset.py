import os
import lmdb
import json
import logging
from torch.utils.data import Dataset, DataLoader
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        logger.info("Initialize LMDB Datasets...")
        self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
            self.lmdb_keys = [key for key, _ in txn.cursor()]
        logger.info("LMDB Datasets Done.")

    def getrawitem(self, index):
        with self.env.begin(write=False) as txn:
            key = self.lmdb_keys[index]
            json_str = txn.get(key).decode("utf-8")
            return json_str

    def process(self, json_str):
        raise NotImplementedError

    def __getitem__(self, index):
        json_str = self.getrawitem(index)
        return self.process(json_str)

    def __len__(self):
        return self.length // 10
