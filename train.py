# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team and Huawei Noah's Ark Lab.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import time
import logging
import os
import random
import sys
import time
import datetime
import shutil
from tools import official_tokenization as tokenization, utils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modeling_nezha import (BertForSequenceClassification, BertForTokenClassification, DocumentBertLSTM,
                            BertForMultiLabelingClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME)
from optimization import BertAdam, warmup_linear
from seqeval.metrics import classification_report



logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s] %(message)s")
logger = logging.getLogger(__name__)


def get_model(args, bert_config, num_labels):
    if args.task_name == "ner":
        return BertForTokenClassification(bert_config, num_labels=num_labels)
    elif args.task_name == "text-clf":
        return BertForSequenceClassification(bert_config, num_labels=num_labels)
    elif args.task_name == "multilabeling":
        if args.encode_document:
            return DocumentBertLSTM(bert_config, args.doc_inner_batch_size, num_labels=num_labels)
        else:
            return BertForMultiLabelingClassification(bert_config, num_labels=num_labels)
    else:
        logger.error("task type not supported!")
        return None


def get_dataloader(args, tokenizer, num_labels, split):
    json_file = os.path.join(args.data_dir, split + ".json")
    if args.task_name == "ner":
        from datasets.ner import NERDataset
        label_map_path = os.path.join(args.data_dir, "label_map")
        dataset = NERDataset(json_file, label_map_path, tokenizer, num_labels=num_labels)
    elif args.task_name == "text-clf":
        pass
    elif args.task_name == "multilabeling":
        from datasets.multilabeling import MultiLabelingDataset
        dataset = MultiLabelingDataset(json_file, tokenizer, num_labels, args.doc_inner_batch_size, args.max_seq_length, args.encode_document)
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    shuffle = True if split == "train" and args.distributed == False else False
    batch_size = args.train_batch_size if split == "train" else args.eval_batch_size
    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        sampler=sampler
                    )
    return len(dataset), dataloader


def train_loop(args, model, train_dataloader, optimizer, num_gpus, global_step):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.cuda() for t in batch)
        if args.encode_document:
            document_batch, label_ids, _ = batch
            loss = model(document_batch, label_ids)
        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)

        if num_gpus > 1 or args.distributed:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                # modify learning rate with special warm up BERT uses
                # if args.fp16 is False, BertAdam is used that handles this automatically
                lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                  args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1


def eval_loop(args, model, eval_dataloader, label_map):
    model.eval()
    eval_func = None
    if args.task_name == "ner":
        pass
    elif args.task_name == "text-clf":
        pass
    elif args.task_name == "multilabeling":
        from evaluation.multilabeling_eval import accuracy_with_thresh as eval_func
        from evaluation.multilabeling_eval import roc_auc

    _all_logits = []
    _all_labels = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.cuda() for t in batch)
        if args.encode_document:
            document_batch, label_ids, _ = batch
            logits = model(document_batch, None)
        else:
            input_ids, input_mask, segment_ids, label_ids = batch
            logits = model(input_ids, segment_ids, input_mask, None)
        _all_logits.append(logits.detach().cpu())
        _all_labels.append(label_ids.detach().cpu())
    all_logits = torch.cat(_all_logits, 0)
    all_labels = torch.cat(_all_labels, 0)
    acc = eval_func(all_logits, all_labels)
    # fpr, tpr, roc_auc_dict = roc_auc(all_logits.numpy(), all_labels.numpy(), len(label_map))
    logger.info("Accuracy: " + str(acc))
    # logger.info("FPR: " + str(fpr))
    # logger.info("TPR: " + str(tpr))
    # logger.info("ROC and AUC: " + str(roc_auc_dict))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--encode_document",
                        action='store_true',
                        help="Whether treat the text as document or not")
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    # trained_model_file
    parser.add_argument("--trained_model_dir", default=None, type=str,
                        help="trained model for eval or predict")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--my_tokenization",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--doc_inner_batch_size",
                        default=5,
                        type=int,
                        help="batch_size in each doc, enabled if encode_document is True")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.no_cuda:
        logger.info("can not train without GPU")
        exit()

    if args.local_rank == -1:
        num_gpus = torch.cuda.device_count()
        args.distributed = False
    else:
        torch.cuda.set_device(args.local_rank)
        num_gpus = 1
        args.distributed = True
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    logger.info("num_gpus: {}, distributed training: {}, 16-bits training: {}".format(
        num_gpus, bool(args.local_rank != -1), args.fp16))
    cudnn.benchmark = True

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if num_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.distributed == False or args.local_rank == 0:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
            logger.warn("Output directory ({}) already exists and is not empty.".format(args.output_dir))
            time.sleep(2)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            # copy vocab.txt from pretrained model dir to output dir
            if args.bert_model:
                shutil.copyfile(os.path.join(args.bert_model, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
            elif args.trained_model_dir and args.trained_model_dir != args.output_dir:
                shutil.copyfile(os.path.join(args.trained_model_dir, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))

    task_name = args.task_name.lower()

    if args.bert_model:
        tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.bert_model, 'vocab.txt'),
                                               do_lower_case=True)
    elif args.trained_model_dir:
        tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.trained_model_dir, 'vocab.txt'),
                                               do_lower_case=True)
    else:
        logger.error("BERT vocab file not set, please check your ber_model_dir or trained_model_dir")

    logger.info('vocab size is %d' % (len(tokenizer.vocab)))

    if args.do_train:
        # label_map {id: label, ...}
        with open(os.path.join(args.data_dir, "label_map")) as f:
            label_map = json.loads(f.read().strip())
            label_map = {int(k):v for k, v in label_map.items()}
        num_labels = len(label_map)

        # copy label_map to output dir
        label_file = os.path.join(args.output_dir, "label_map_training.txt")
        shutil.copyfile(
            os.path.join(args.data_dir, "label_map"),
            os.path.join(args.output_dir, "label_map")
        )

        # label_map_reverse {label: id, ...}
        label_map_reverse = {v: k for k, v in label_map.items()}

        # TODO add label_map_reverse into nerdataset
        num_examples, train_dataloader = get_dataloader(args, tokenizer, num_labels, "train")

        num_train_optimization_steps = int(
            num_examples / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    else:
        # TODO not train...
        # do some other things
        pass

    if args.trained_model_dir:
        logger.info('init nezha model from user fine-tune model...')
        config = BertConfig(os.path.join(args.trained_model_dir, 'bert_config.json'))
        model = get_model(args, config, num_labels=num_labels)
        model.load_state_dict(torch.load(os.path.join(args.trained_model_dir, WEIGHTS_NAME)))
    elif args.bert_model:
        logger.info('init nezha model from original pretrained model...')
        bert_config = BertConfig.from_json_file(os.path.join(args.bert_model, 'bert_config.json'))
        model = get_model(args, bert_config, num_labels=num_labels)
        utils.torch_show_all_params(model)
        utils.torch_init_model(model, os.path.join(args.bert_model, 'pytorch_model.bin'))

    if args.fp16:
        model.half()

    if args.distributed:
        model.cuda(args.local_rank)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank])

        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        # model = DDP(model)
    elif num_gpus > 1:
        model.cuda()
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            # from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("================ start training on train set ===================")
        num_epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            # train loop in one epoch
            train_loop(args, model, train_dataloader, optimizer, num_gpus, global_step)

            # begin to evaluate
            logger.info("================ running evaluation on dev set ===================")
            _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "dev")
            eval_loop(args, model, eval_dataloader, label_map)

            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

            num_epoch += 1

    model.to(device)
    logger.info('%s' % str(args.do_eval))
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("================ running evaluation on dev set ===================")
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "dev")
        eval_loop(args, model, eval_dataloader, label_map)

    if args.do_test:
        logger.info("================ running evaluation in test set ===================")
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "test")
        eval_loop(args, model, eval_dataloader, label_map)

if __name__ == "__main__":
    main()
