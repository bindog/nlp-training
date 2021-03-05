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
from stopwordsiso import stopwords


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

lac = LAC(mode='seg')
all_stopwords = stopwords(["en", "zh"])


def cut_words(raw_dict):
    raw_json_dict = json.loads(raw_dict["raw_json_text"])
    seg_words = lac.run(raw_json_dict["text"])
    # remove stop words
    seg_words_clean = [w for w in seg_words if w not in all_stopwords]
    raw_json_dict["words"] = seg_words_clean
    return raw_json_dict


def create(data_home, split="train", cut=False, num_workers=40):
    r"""Convert json format training files to LMDB Database format
        With the ProcessPoolExecutor we can enjoy the ultimate high speed

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
                # leave json parsing into sub-process
                "raw_json_text": line.strip()
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
            future_to_nid[future] = nid
            future.add_done_callback(fcallback)


    lmdb_path = os.path.join(data_home, split + ".lmdb")
    logger.info("Create LMDB Database with path: " + str(lmdb_path))
    env = lmdb.open(lmdb_path, map_size=1 << 40)

    with open(json_path, "r") as f, env.begin(write=True) as txn:
        count = 0
        for nid, raw_dict in all_raw_dict.items():
            serialized_string = pickle.dumps(raw_dict)
            key = "{:08}".format(count)
            txn.put(key.encode(), serialized_string)
            count += 1
            if count % 10000 == 0:
                logger.info("LMDB Database create processed {:08} lines...".format(count))

    logger.info("LMDB Database create done.")


if __name__ == "__main__":
    fire.Fire()
