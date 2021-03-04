import os
import json
import time
import lmdb
import fire
import pickle
import logging
import traceback
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from LAC import LAC


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

lac = LAC(mode='seg')


def cut_words(raw_dict):
    raw_json_dict = json.loads(raw_dict["raw_text"])  # raw json text
    seg_words = lac.run(raw_json_dict["text"])
    # TODO remove stop words...
    raw_dict["words"] = seg_words
    return raw_dict


def create(data_home, split="train", cut=False, num_workers=40):
    r"""Convert json format training files to LMDB format

        Args:
            data_home: json files data home
            split: train dev/val test
            cut: cut words or not (for chinese text only)
            num_workers: num of workers for ProcessPoolExecutor
        Returns:
            None
    """
    json_path = os.path.join(data_home, split + ".json")

    all_raw_dict = {}
    nid = 0
    logger.info("reading raw input file...")
    with open(json_path, "r") as f:
        for line in f:
            all_raw_dict[nid] = {
                "id": nid,
                "raw_text": line.strip()  # raw json text
            }
            nid += 1

    logger.info("cutting words...")
    with tqdm(total=len(all_raw_dict)) as pbar, ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_nid = {}
        def fcallback(future):
            nid = future_to_nid[future]
            pbar.update(1)
            all_raw_dict[nid] = future.result()

        for nid, raw_dict in all_raw_dict.items():
            future = executor.submit(cut_words, raw_dict)
            future.add_done_callback(fcallback)
            future_to_nid[future] = nid


    logger.info("create lmdb database...")
    lmdb_path = os.path.join(data_home, split + ".lmdb")
    env = lmdb.open(lmdb_path, map_size=1 << 40)
    # lmdb_keys = []

    with open(json_path, "r") as f, env.begin(write=True) as txn:
        count = 0
        for nid, raw_dict in all_raw_dict.items():
            json_str = json.dumps(raw_dict, ensure_ascii=False)
            key = "{:08}".format(count)
            txn.put(key.encode(), json_str.encode())
            # lmdb_keys.append(key)
            count += 1
            if count % 10000 == 0:
                logger.info("lmdb create processed {:08} lines...".format(count))

    # pkl_path = os.path.join(data_home, split + "_keys.pkl")
    # with open(pkl_path, "wb") as f:
    #     pickle.dump(lmdb_keys, f)

    logger.info("lmdb create done.")


if __name__ == "__main__":
    fire.Fire()
