import os
import json
import logging
import mmap
from itertools import chain, repeat, dropwhile
from multiprocessing import Pool, Process, current_process

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")
logger = logging.getLogger(__name__)


def rindex(lst, item):
    def index_ne(x):
        return lst[x] != item
    try:
        return next(dropwhile(index_ne, reversed(range(len(lst)))))
    except StopIteration:
        # raise ValueError("rindex(lst, item): item not in list")
        return -1


def process_chunk(chunk, tokenizer, label_map_reverse, num_labels=18, max_seq_length=128):
    punctuation = ["，", "。", "：", "？", "！", "、", "；"]

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    part = chunk.split(b'\n')
    for p in part:
        line = p.decode("utf-8")
        if len(line) < 6:
            continue
        info = json.loads(line.strip())
        raw_text = info["raw"]
        words = info["word"]
        entity_list = info["entity"]

        raw_label_ids = [label_map_reverse["O"]] * len(raw_text)
        for entity in entity_list:
            s, e, st, et = entity

            raw_label_ids[s] = label_map_reverse[st]
            raw_label_ids[s+1:e] = [label_map_reverse[et]] * (e-s-1)

        tokens_a = []
        for char in raw_text:
            _token = tokenizer.tokenize(char)
            if len(_token) == 1:
                tokens_a.extend(_token)
            else:
                tokens_a.append("[UNK]")

        assert len(tokens_a) == len(raw_label_ids), "tokens length must equals labels length"

        sindex = 0
        eindex = 0
        end_flag = False
        while sindex < len(raw_text):
            _eindex = sindex + max_seq_length - 2
            if _eindex >= len(raw_text):
                if len(raw_text[sindex:_eindex]) < max_seq_length // 2:
                    break
                else:
                    end_flag = True
            # find end by punctuation
            p_list = []
            for p in punctuation:
                p_index = raw_text[sindex:_eindex].rfind(p)
                p_list.append(p_index)
            if max(p_list) > 0:
                eindex = sindex + max(p_list)
            else:
                # find end by label
                label_o_index = rindex(raw_label_ids[sindex:_eindex], label_map_reverse["O"])
                if label_o_index > 0:
                    eindex = sindex + label_o_index
                else:
                    # logger.info("can not found split point, directly split")
                    eindex = _eindex

            # convert and padding data
            tokens = ["[CLS]"] + tokens_a[sindex:eindex] + ["[SEP]"]
            label_ids = [label_map_reverse["[CLS]"]] + raw_label_ids[sindex:eindex] + [label_map_reverse["[SEP]"]]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            label_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)
            all_label_ids.append(label_ids)

            if end_flag:
                break

            sindex = eindex

    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


def split_chunks(filename, grain=10000):
    '''
    using mmap and seek identifier to split whole file into small chunks
    then use multiprocessing to speed up process
    '''
    f = open(filename, "r+b")
    s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    start = 0
    prefix = b'{"id": '
    chunks = []

    cursor = 0
    while True:
        initial = s.find(prefix + str(start).encode("utf-8"), cursor)
        if initial == -1:
            break
        cursor = initial + 66
        final = s.find(prefix + str(start + grain).encode("utf-8"), cursor)
        start += grain
        if final == -1:
            chunks.append(s[initial:])
            break
        chunks.append(s[initial:final])

    if len(chunks) == 0:
        chunks.append(s[:])

    logger.info("total chunks: " + str(len(chunks)))
    s.close()
    f.close()
    return chunks


class NERDataset(Dataset):
    def __init__(self, json_path, label_map_path, tokenizer, num_labels=None, max_seq_length=128):
        """Initiate MultiLabelingDataset dataset.
        Arguments:
            json_path:
            num_labels:
            tokenization:
        """
        super(NERDataset, self).__init__()

        with open(label_map_path, "r") as f:
            label_map = json.loads(f.read().strip())
            self.label_map_reverse = {v:int(k) for k, v in label_map.items()}

        self.num_labels = num_labels

        assert self.num_labels == len(self.label_map_reverse), "num_labels should equals to label_map"

        logger.info("prepare multilabeling dataset from: " + json_path)
        logger.info("number of labels: " + str(self.num_labels))

        # use mmap and seek to split the huge data file into small chunks
        chunks = split_chunks(json_path, grain=4000)

        # multiprocessing parsing chunks
        pool = Pool(16)
        results = pool.starmap(
            process_chunk,
            zip(chunks, repeat(tokenizer), repeat(self.label_map_reverse), repeat(self.num_labels), repeat(max_seq_length))
        )

        all_results = []
        for parts in zip(*results):
            all_results.append(list(chain(*parts)))

        self.all_input_ids = torch.tensor(all_results[0], dtype=torch.long)
        self.all_input_mask = torch.tensor(all_results[1], dtype=torch.long)
        self.all_segment_ids = torch.tensor(all_results[2], dtype=torch.long)
        self.all_label_ids = torch.tensor(all_results[3], dtype=torch.long)
        logger.info("multilabeling dataset ready...")

    def __getitem__(self, i):
        return self.all_input_ids[i], self.all_input_mask[i], self.all_segment_ids[i], self.all_label_ids[i]

    def __len__(self):
        return len(self.all_input_ids)
