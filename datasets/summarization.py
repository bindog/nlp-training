import os
import json
import logging
import mmap
from itertools import chain, repeat
from multiprocessing import Pool, Process, current_process

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def process_chunk(chunk, tokenizer, max_source_length=1024, max_target_length=56, crosslingual=False):
    all_input_ids = []
    all_attention_mask = []
    all_label_ids = []

    part = chunk.split(b'\n')
    for p in part:
        line = p.decode("utf-8")
        if len(line) < 6:
            continue
        info = json.loads(line.strip())
        raw_text = info["text"]
        if crosslingual:
            # sl_summary = info["sl_summary"]
            # target language summary
            summary = info["tl_summary"]
        else:
            summary = info["summary"]

        source_tokenized = tokenizer.batch_encode_plus(
                [raw_text], max_length=max_source_length, truncation=True, padding="max_length", return_tensors=None
        )
        target_tokenized = tokenizer.batch_encode_plus(
                [summary], max_length=max_target_length, truncation=True, padding="max_length", return_tensors=None
        )
        all_input_ids.append(source_tokenized["input_ids"][0])
        all_attention_mask.append(source_tokenized["attention_mask"][0])
        all_label_ids.append(target_tokenized["input_ids"][0])

    return all_input_ids, all_attention_mask, all_label_ids


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


class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, max_source_length=1024, max_target_length=56, crosslingual=False):
        """Initiate Textclf Dataset dataset.
        Arguments:
            json_path: dataset json file
            tokenizer: BERT tokenizer
            max_source_length: the max source sequence length which BERT seq2seq supports
            max_target_length: the max target sequence length which BERT seq2seq supports
        """
        super(SummarizationDataset, self).__init__()

        logger.info("prepare summarization dataset from: " + json_path)

        # use mmap and seek to split the huge data file into small chunks
        chunks = split_chunks(json_path, grain=4000)

        # multiprocessing parsing chunks
        pool = Pool(16)
        results = pool.starmap(
            process_chunk,
            zip(
                chunks,
                repeat(tokenizer),
                repeat(max_source_length),
                repeat(max_target_length),
                repeat(crosslingual),
            )
        )

        all_results = []
        for parts in zip(*results):
            all_results.append(list(chain(*parts)))
        self.all_input_ids = torch.tensor(all_results[0], dtype=torch.long)
        self.all_attention_mask = torch.tensor(all_results[1], dtype=torch.long)
        self.all_label_ids = torch.tensor(all_results[2], dtype=torch.long)
        logger.info("summarization dataset ready...")

    def __getitem__(self, i):
        return {
            "input_ids": self.all_input_ids[i],
            "attention_mask": self.all_attention_mask[i],
            "labels": self.all_label_ids[i]
        }

    def __len__(self):
        return len(self.all_label_ids)
