import os
import json
import logging
import mmap
from itertools import chain, repeat
from multiprocessing import Pool, Process, current_process

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")
logger = logging.getLogger(__name__)


def process_chunk(chunk, tokenizer, num_labels=11, max_seq_length=128):
    _all_input_ids = []
    _all_input_mask = []
    _all_segment_ids = []
    _all_label_ids = []
    part = chunk.split(b'\n')
    for p in part:
        line = p.decode("utf-8")
        if len(line) < 6:
            continue
        info = json.loads(line.strip())
        raw_text = info["raw"]
        multilabel = info["multilabel"]

        tokens_a = tokenizer.tokenize(raw_text)
        # TODO we can split the text into more training examples
        # ...
        s_index = 0
        e_index = max_seq_length - 2
        while s_index == 0 or len(tokens_a) - s_index > max_seq_length // 2:
            tokens_a_i = tokens_a[s_index:e_index]
            tokens = ["[CLS]"] + tokens_a_i + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_ids = [1 if i in multilabel else 0 for i in range(num_labels)]
            _all_input_ids.append(input_ids)
            _all_input_mask.append(input_mask)
            _all_segment_ids.append(segment_ids)
            _all_label_ids.append(label_ids)

            s_index = e_index
            e_index = s_index + max_seq_length - 2

    return _all_input_ids, _all_input_mask, _all_segment_ids, _all_label_ids


def split_chunks(filename, grain=10000):
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


class MultiLabelingDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, num_labels, tokenizer, max_seq_length=128):
        """Initiate MultiLabelingDataset dataset.
        Arguments:
            json_path:
            num_labels:
            tokenization:
        """
        super(MultiLabelingDataset, self).__init__()
        self.num_labels = num_labels

        logger.info("prepare multilabeling dataset from: " + json_path)
        logger.info("number of labels: " + str(num_labels))

        # use mmap and seek to split the huge data file into small chunks
        chunks = split_chunks(json_path, grain=4000)

        # multiprocessing parsing chunks
        pool = Pool(16)
        results = pool.starmap(
            process_chunk,
            zip(chunks, repeat(tokenizer), repeat(self.num_labels), repeat(max_seq_length))
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
