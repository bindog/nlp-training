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

FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR",
    "cs_CZ",
    "de_DE",
    "en_XX",
    "es_XX",
    "et_EE",
    "fi_FI",
    "fr_XX",
    "gu_IN",
    "hi_IN",
    "it_IT",
    "ja_XX",
    "kk_KZ",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "my_MM",
    "ne_NP",
    "nl_XX",
    "ro_RO",
    "ru_RU",
    "si_LK",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]

# https://huggingface.co/transformers/model_doc/mbart.html
# translation mbart model train
# model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels) #forward

# generation
# article = "text ..."
# batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], src_lang="en_XX")
# translated_tokens = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
# translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def process_chunk(chunk, tokenizer, max_source_length=1024, max_target_length=1024):
    all_input_ids = []
    all_decoder_input_ids = []
    all_labels_ids = []

    part = chunk.split(b'\n')
    for p in part:
        line = p.decode("utf-8")
        if len(line) < 6:
            continue
        info = json.loads(line.strip())
        raw_text = info["text"]
        translation_text = info["translation"]
        src_lang = info["src_lang"]
        tgt_lang = info["tgt_lang"]

        batch = tokenizer.prepare_seq2seq_batch(
            raw_text, src_lang=src_lang, tgt_texts=translation_text, tgt_lang=tgt_lang,
            max_length=max_source_length, max_target_length=max_target_length, padding="max_length", return_tensors=None
        )
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        all_input_ids.append(input_ids)
        all_labels_ids.append(labels)

    return all_input_ids, all_labels_ids


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


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, max_source_length=1024, max_target_length=56, crosslingual=False):
        """Initiate Textclf Dataset dataset.
        Arguments:
            json_path: dataset json file
            tokenizer: BERT tokenizer
            max_source_length: the max source sequence length which BERT seq2seq supports
            max_target_length: the max target sequence length which BERT seq2seq supports
        """
        super(TranslationDataset, self).__init__()

        logger.info("prepare translation dataset from: " + json_path)

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
                # repeat(crosslingual),
            )
        )

        all_results = []
        for parts in zip(*results):
            all_results.append(list(chain(*parts)))
        self.all_input_ids = torch.tensor(all_results[0], dtype=torch.long)
        self.all_label_ids = torch.tensor(all_results[1], dtype=torch.long)
        logger.info("translation dataset ready...")


    def __getitem__(self, i):
        return {
            "input_ids": self.all_input_ids[i],
            "labels": self.all_label_ids[i]
        }

    def __len__(self):
        return len(self.all_label_ids)
