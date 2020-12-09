from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
import json
import datetime
import shutil
import numpy as np
from bidict import bidict
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from models.modeling_nezha import (NeZhaForSequenceClassification, NeZhaForTokenClassification, NeZhaForTagClassification,
                                   NeZhaForDocumentClassification, NeZhaForDocumentTagClassification)


from datasets.textclf import encode_single_document

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


class SummarizationInferenceService(object):
    def __init__(self, model_dir, **kwargs):
        from models.tokenization_mbart import MBartTokenizer
        from models.modeling_mbart import MBartForConditionalGeneration
        self.model = MBartForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = MBartTokenizer.from_pretrained(model_dir)
        self.model.cuda()
        self.model.eval()

    def input_preprocess(self):
        pass

    def inference(self, text):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt')
        with torch.no_grad():
            summary_ids = self.model.generate(inputs.input_ids.cuda(), num_beams=4, max_length=50, early_stopping=True)
            summary_text = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        print("debug summary:", summary_text)
        return "".join(summary_text)


class TextclfInferenceService(object):
    def __init__(self, model_dir, encode_document=False, doc_inner_batch_size=5, tag=False, **kwargs):
        with open(os.path.join(model_dir, "label_map")) as f:
            label_map = json.loads(f.read().strip())
            self.label_map = {int(k):v for k, v in label_map.items()}
        num_labels = len(self.label_map)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.encode_document = encode_document
        self.tag = tag

        if encode_document:
            self.doc_inner_batch_size = doc_inner_batch_size
            if self.tag:
                self.model = NeZhaForDocumentTagClassification.from_pretrained(model_dir, doc_inner_batch_size=doc_inner_batch_size, num_labels=num_labels)
            else:
                self.model = NeZhaForDocumentClassification.from_pretrained(model_dir, doc_inner_batch_size=doc_inner_batch_size, num_labels=num_labels)
        else:
            if self.tag:
                self.model = NeZhaForTagClassification.from_pretrained(model_dir, num_labels=num_labels)
            else:
                self.model = NeZhaForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        self.model.eval()
        self.model.cuda()

    def get_label_map(self):
        return self.label_map

    def input_preprocess(self, text, max_seq_length=128):
        if self.encode_document:
            if len(text) > (max_seq_length - 2) * self.doc_inner_batch_size:
                logger.warn("text too long, we only take the {} words in the beginning...".format((max_seq_length - 2) * self.doc_inner_batch_size))
            document_compose, num_seq_in_doc = encode_single_document(text, self.tokenizer, self.doc_inner_batch_size, max_seq_length)
            document_compose = torch.tensor(document_compose, dtype=torch.long).unsqueeze(0)
            return document_compose,
        else:
            if len(text) > max_seq_length - 2:
                logger.warn("text too long, we only take the {} words in the beginning...".format(max_seq_length))

            tokens_a = self.tokenizer.tokenize(text)
            tokens_a = tokens_a[:max_seq_length - 2]
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            return input_ids, input_mask, segment_ids

    def parse_label(self, preds):
        # single batch inference
        if self.tag:
            idx = preds.nonzero()[1]
            labels = [self.label_map[i] for i in idx]
            return labels
        else:
            return self.label_map[preds]

    def inference(self, text, thresh=0.5, parse_label=True):
        # single batch inference
        inputs = self.input_preprocess(text)
        with torch.no_grad():
            inputs = (t.cuda() for t in inputs)
            logits = self.model(*inputs)
            print("debug logits:", logits)
            if self.tag:
                preds = (logits.detach().cpu().sigmoid() > thresh).byte().numpy()
            else:
                _, preds = torch.max(logits.detach().cpu(), 1)
                preds = preds.byte().numpy()[0]
            if parse_label:
                return self.parse_label(preds)
            else:
                return preds


class NERInferenceService(object):
    def __init__(self, model_dir, **kwargs):
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                        "B-PRO", "I-PRO", "B-JOB", "I-JOB", "B-TIME", "I-TIME",
                        "B-COM", "I-COM", "X", "[CLS]", "[SEP]"]
        self.label_map = {label: i  for i, label in enumerate(label_list,1)}
        num_labels = len(label_list) + 1
        self.mapping_dict = {
            "PER": "人名",
            "LOC": "地名",
            "ORG": "组织",
            "JOB": "组织",
            "PRO": "产品",
            "TIME": "时间",
            "COM": "公司"
        }
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = NeZhaForTokenClassification.from_pretrained(model_dir, num_labels=num_labels)
        self.model.eval()
        self.model.cuda()

    def input_preprocess(self, text, max_seq_length=128):
        if len(text) > max_seq_length:
            # split or other way?
            pass
        else:
            tokens = []
            valid = []
            for i, word in enumerate(text):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            ntokens.append("[SEP]")
            segment_ids.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_mask = torch.tensor([input_mask], dtype=torch.long)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long)
            return input_ids, input_mask, segment_ids

    def parse_label(self, text, label):
        if label[0] != "[CLS]":
            print("wrong result in first label")
            return None

        entity_list = []
        index = 0
        entity_type = None
        entity_start = None
        entity_end = None
        for i, tag in enumerate(label[1:]):
            if tag in ["B-PER", "B-ORG", "B-LOC", "B-PRO", "B-JOB", "B-TIME", "B-COM"]:
                if entity_type is not None:
                    entity_end = index
                    entity_list.append((entity_start, entity_end, text[entity_start: entity_end], entity_type))
                entity_type = tag[2:]
                entity_start = index
            elif tag in ["I-PER", "I-ORG", "I-LOC", "I-PRO", "I-JOB", "I-TIME", "I-COM"]:
                if tag[2:] != entity_type:
                    print("wrong result in middle parsing...")
            else:
                if entity_type is not None:
                    entity_end = index
                    entity_list.append((entity_start, entity_end, text[entity_start: entity_end], entity_type))
                    entity_type = None
            index += 1
        return entity_list

    def inference(self, text):
        input_ids, input_mask, segment_ids = self.input_preprocess(text)
        with torch.no_grad():
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()
            logits = self.model(input_ids, segment_ids, input_mask)
        logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label = [self.label_map.inverse[i] for i in logits[0]]
        entity_list = self.parse_label(text, label)

        result = []
        for r in entity_list:
            rj = {}
            rj["entity"] = r[2]
            rj["type"] = self.mapping_dict[r[3]]
            rj["offset"] = [r[0], r[1]]
            result.append(rj)
        return result


class AttitudeInferenceService(object):
    def __init__(self, model_dir, **kwargs):
        from models.tokenization_t5 import T5Tokenizer
        from models.modeling_mt5 import MT5ForConditionalGeneration
        self.model = MT5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model.cuda()
        self.model.eval()

        # four classes
        self.cloze_words = bidict({
            0: "中立",
            1: "负面",
            2: "恶意",
            3: "憎恨",
        })
        self.cloze_length = 2
        self.cloze_context = {
                        "position": "head",
                        "pin": "<extra_id_0>",
                        "prefix": "这是一篇对华态度",
                        "suffix": "的报道。"
                    }

    def input_preprocess(self, text):
        processed_text = self.cloze_context["prefix"] + " " + \
                    self.cloze_context["pin"] + " " + \
                    self.cloze_context["suffix"] + text
        inputs = self.tokenizer([processed_text], max_length=1024, return_tensors='pt')
        return inputs.input_ids

    def inference(self, text):
        input_ids = self.input_preprocess(text)
        with torch.no_grad():
            attitude_ids = self.model.generate(input_ids.cuda(), num_beams=4, max_length=5, early_stopping=True)
            raw_attitude_text = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in attitude_ids]
        print("debug attitude:", raw_attitude_text)
        attitude_text = "".join(raw_attitude_text)[-self.cloze_length:]
        return self.cloze_words.inv[attitude_text]


class XSummaryInferenceService(object):
    def __init__(self, model_dir, **kwargs):
        from models.tokenization_t5 import T5Tokenizer
        from models.modeling_mt5 import MT5ForConditionalGeneration
        self.model = MT5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        self.model.cuda()
        self.model.eval()
        self.prefix = "英文摘要成中文: "

    def input_preprocess(self, text):
        processed_text = self.prefix + text
        inputs = self.tokenizer([processed_text], max_length=1024, return_tensors='pt')
        return inputs.input_ids

    def inference(self, text):
        input_ids = self.input_preprocess(text)
        summary_ids = self.model.generate(input_ids.cuda(), num_beams=5, max_length=30, early_stopping=True, use_cache=True)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("debug summary: ", summary_text)
        return summary_text
