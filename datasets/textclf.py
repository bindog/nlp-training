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


def encode_single_document(document, tokenizer, max_seq_per_doc=5, max_seq_length=128):
    """
    Args:
        document: long document of raw text
        tokenizer: BERT tokenizer
        max_seq_per_doc: this should be equal to the batch size of bert model
        max_seq_length: max_seq_length of BERT model
    Returns:
        output: tensor of shape [max_seq_per_doc, 3, max_seq_length], including input_ids, segment_ids, input_mask
        num_seq_in_doc: real number of sequences in output
    """
    tokenized_document = tokenizer.tokenize(document)

    place_holder = ([0] * max_seq_length, [0] * max_seq_length, [0] * max_seq_length)
    output = [place_holder] * max_seq_per_doc

    num_seq_in_doc = 0
    for seq_index, i in enumerate(range(0, len(tokenized_document), (max_seq_length - 2))):
        if seq_index == max_seq_per_doc:
            break

        raw_tokens = tokenized_document[i: i + (max_seq_length - 2)]

        tokens = ["[CLS]"] + raw_tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        output[seq_index] = (input_ids, segment_ids, input_mask)
        num_seq_in_doc = seq_index
    return output, num_seq_in_doc + 1


def process_chunk(chunk, tokenizer, num_labels=11, max_seq_per_doc=5, max_seq_length=128, encode_document=False, tag=False):
    if encode_document:
        all_document_compose = []
        all_num_seq_list= []
        all_label_ids = []
    else:
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
        raw_text = info["text"]

        if tag:  # tag dataset
            category = info["tag"]  # int list
        else:
            category = info["category"]  # int

        if encode_document:
            output, num_seq_in_doc = encode_single_document(raw_text, tokenizer, max_seq_per_doc, max_seq_length)
            all_document_compose.append(output)
            all_num_seq_list.append(num_seq_in_doc)
            if tag:  # tag dataset
                # the label_ids will become one-hot style
                label_ids = [1 if i in category else 0 for i in range(num_labels)]
                all_label_ids.append(label_ids)
            else:
                all_label_ids.append(category)
        else:
            tokens_a = tokenizer.tokenize(raw_text)
            # split a long text into small text parts
            # they all share the same label
            s_index = 0
            e_index = max_seq_length - 2
            while s_index == 0 or len(tokens_a) - s_index > max_seq_length // 2:
                tokens_a_i = tokens_a[s_index:e_index]
                tokens = ["[CLS]"] + tokens_a_i + ["[SEP]"]
                segment_ids = [0] * len(tokens)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                # zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                if tag:  # tag dataset
                    label_ids = [1 if i in category else 0 for i in range(num_labels)]
                    all_label_ids.append(label_ids)
                else:
                    all_label_ids.append(category)

                all_input_ids.append(input_ids)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                all_label_ids.append(label_ids)

                s_index = e_index
                e_index = s_index + max_seq_length - 2

    if encode_document:
        return all_document_compose, all_label_ids, all_num_seq_list
    else:
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


class TextclfDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, num_labels=None, max_seq_per_doc=6, max_seq_length=128, encode_document=False, tag=False):
        """Initiate Textclf Dataset dataset.
        Arguments:
            json_path: dataset json file
            tokenizer: BERT tokenizer
            num_labels: number of labels
            max_seq_per_doc: enabled if encode_document is True, the max inner batch_size of a document
            max_seq_length: the max sequence length which BERT supports
            encode_document: whether treat the text as a document
            tag: the labels will be multi_label if True, else just a normal label
        """
        super(TextclfDataset, self).__init__()
        self.num_labels = num_labels
        self.encode_document = encode_document
        self.tag = tag

        logger.info("prepare textclf dataset from: " + json_path)
        logger.info("number of labels: " + str(num_labels) + "\tmultilabel tagging: " + str(tag))
        if self.encode_document:
            max_doc_length = (max_seq_length - 2) * max_seq_per_doc
            logger.info("encode document enabled, the longest length of a document can be: " + str(max_doc_length))

        # use mmap and seek to split the huge data file into small chunks
        chunks = split_chunks(json_path, grain=4000)

        # multiprocessing parsing chunks
        pool = Pool(16)
        results = pool.starmap(
            process_chunk,
            zip(
                chunks,
                repeat(tokenizer),
                repeat(self.num_labels),
                repeat(max_seq_per_doc),
                repeat(max_seq_length),
                repeat(encode_document),
                repeat(tag)
            )
        )

        all_results = []
        for parts in zip(*results):
            all_results.append(list(chain(*parts)))
        if encode_document:
            self.all_document_compose = torch.tensor(all_results[0], dtype=torch.long)
            self.all_label_ids = torch.tensor(all_results[1], dtype=torch.long)
            self.all_num_seq_list = torch.tensor(all_results[2], dtype=torch.long)
        else:
            self.all_input_ids = torch.tensor(all_results[0], dtype=torch.long)
            self.all_input_mask = torch.tensor(all_results[1], dtype=torch.long)
            self.all_segment_ids = torch.tensor(all_results[2], dtype=torch.long)
            self.all_label_ids = torch.tensor(all_results[3], dtype=torch.long)
        logger.info("textclf dataset ready...")

    def __getitem__(self, i):
        if self.encode_document:
            return self.all_document_compose[i], self.all_label_ids[i]
        else:
            return self.all_input_ids[i], self.all_input_mask[i], self.all_segment_ids[i], self.all_label_ids[i]

    def __len__(self):
        return len(self.all_label_ids)
