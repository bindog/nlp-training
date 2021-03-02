import os
import json
import lmdb
import pickle
import logging
import argparse


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def json_to_lmdb(data_home, split="train"):
    json_path = os.path.join(data_home, split + ".json")
    lmdb_path = os.path.join(data_home, split + ".lmdb")
    env = lmdb.open(lmdb_path, map_size=1 << 40)
    lmdb_keys = []

    with open(json_path, "r") as f, env.begin(write=True) as txn:
        count = 0
        for line in f:
            json_str = line.strip()
            key = "{:08}".format(count)
            txn.put(key.encode(), json_str.encode())
            lmdb_keys.append(key)
            count += 1
            if count % 10000 == 0:
                logger.info("lmdb create processed {:08} lines...".format(count))

    pkl_path = os.path.join(data_home, split + "_keys.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(lmdb_keys, f)

    logger.info("lmdb create done.")


def test_read_lmdb(data_home, split="train"):
    lmdb_path = os.path.join(data_home, split + ".lmdb")
    pkl_path = os.path.join(data_home, split + "_keys.pkl")
    env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with open(pkl_path, "rb") as f:
        lmdb_keys = pickle.load(f, encoding="bytes")
    with env.begin(write=False) as txn:
        total_len = txn.stat()["entries"]
        for key in lmdb_keys:
            print(key)
            _key = key.encode() if not isinstance(key, bytes) else key
            json_str = txn.get(_key).decode("utf-8")
            line_dict = json.loads(json_str)
    print("totali", total_len)


if __name__ == "__main__":
    json_to_lmdb("/mnt/dl/public/public_datasets/wmt19_en_ru", "train")
    # test_read_lmdb("/mnt/dl/public/public_datasets/wmt19_en_ru", "val")
