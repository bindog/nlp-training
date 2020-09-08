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

import os
import argparse
import random
import sys
import time
import datetime
import shutil
import csv
import json
import time
import logging
import wandb

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)

from packaging import version
from tools import official_tokenization as tokenization, utils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from models.modeling_nezha import (NeZhaForSequenceClassification, NeZhaForTokenClassification,
                            NeZhaForDocumentClassification, NeZhaForDocumentTagClassification,
                            NeZhaForTagClassification, NeZhaConfig, WEIGHTS_NAME, CONFIG_NAME)

from optimization import AdamW, get_linear_schedule_with_warmup

# check fp16 settings
_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    try:
        from apex import amp
        _use_apex = True
    except ImportError:
        _use_apex = False
        logger.error("apex installation is broken and your pytorch version below 1.6, you CAN NOT use fp16!")
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


def get_split_path(data_dir, split):
    if split == "val" or split == "dev" or split == "valid":
        c_list = ["val", "dev", "valid"]
        for c in c_list:
            json_path = os.path.join(data_dir, c + ".json")
            if os.path.exists(json_path):
                return json_path
    return os.path.join(data_dir, split + ".json")


def get_tokenizer(args):
    if args.model_name == "nezha":
        if args.bert_model:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.bert_model, 'vocab.txt'),
                                                   do_lower_case=True)
        elif args.trained_model_dir:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.trained_model_dir, 'vocab.txt'),
                                                   do_lower_case=True)
        else:
            logger.error("BERT vocab file not set, please check your ber_model_dir or trained_model_dir")
        logger.info('vocab size is %d' % (len(tokenizer.vocab)))
    elif args.model_name == "longformer":
        from models.tokenization_longformer import LongformerTokenizer
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    elif args.model_name == "bart":
        from models.tokenization_mbart import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    else:
        logger.error("can not find the proper tokenizer type...")
    return tokenizer


def get_model(args, bert_config, num_labels):
    if args.task_name == "ner":
        return NeZhaForTokenClassification(bert_config, num_labels=num_labels)
    elif args.task_name == "textclf":
        if args.model_name == "longformer":
            from models.configuration_longformer import LongformerConfig
            from models.modeling_longformer import LongformerForSequenceClassification
            config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
            config.num_labels = num_labels
            model = LongformerForSequenceClassification(config)
            model.from_pretrained("allenai/longformer-base-4096")
            model.freeze_encoder()
            model.unfreeze_encoder_last_layers()
            return model
        elif args.model_name == "nezha":
            if args.encode_document:
                model = NeZhaForDocumentClassification(bert_config, args.doc_inner_batch_size, num_labels=num_labels)
                if args.freeze_encoder:
                    model.freeze_encoder()
                    model.unfreeze_encoder_last_layers()
                return model
            else:
                return NeZhaForSequenceClassification(bert_config, num_labels=num_labels)
        else:
            logger.error("can not find the proper model type...")
            return None
    elif args.task_name == "tag":
        if args.encode_document:
            model = NeZhaForDocumentTagClassification(bert_config, args.doc_inner_batch_size, num_labels=num_labels)
            if args.freeze_encoder:
                model.freeze_encoder()
                model.unfreeze_encoder_last_layers()
            return model
        else:
            return NeZhaForTagClassification(bert_config, num_labels=num_labels)
    elif args.task_name == "summary":
        from models.modeling_mbart import MBartForConditionalGeneration
        gradient_checkpointing_flag = True if args.gradient_checkpointing else False
        if gradient_checkpointing_flag:
            logger.info("gradient checkpointing enabled")
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", gradient_checkpointing=gradient_checkpointing_flag)
        if args.freeze_encoder:
            model.freeze_encoder()
            model.unfreeze_encoder_last_layers()
        return model
    else:
        logger.error("task type not supported!")
        return None


def get_optimizer_and_scheduler(args, model, num_training_steps):
    """
    Setup the optimizer and the learning rate scheduler.
    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler


def get_dataloader(args, tokenizer, num_labels, split):
    json_file = get_split_path(args.data_dir, split)
    if args.task_name == "ner":
        from datasets.ner import NERDataset
        label_map_path = os.path.join(args.data_dir, "label_map")
        dataset = NERDataset(json_file, label_map_path, tokenizer, num_labels=num_labels)
    elif args.task_name == "textclf":
        longformer = True if args.model_name == "longformer" else False
        from datasets.textclf import TextclfDataset
        dataset = TextclfDataset(json_file, tokenizer, num_labels, args.doc_inner_batch_size, args.max_seq_length, args.encode_document, longformer)
    elif args.task_name == "tag":
        from datasets.textclf import TextclfDataset
        dataset = TextclfDataset(json_file, tokenizer, num_labels, args.doc_inner_batch_size, args.max_seq_length, args.encode_document, longformer, tag=True)
    elif args.task_name == "summary":
        from datasets.summarization import SummarizationDataset
        dataset = SummarizationDataset(json_file, tokenizer)
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


def train_loop(args, model, train_dataloader, optimizer, lr_scheduler, num_gpus, epoch, scaler=None):
    model.train()
    p = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(p):
        inputs = {k: v.cuda() for k, v in batch.items()}

        if args.fp16 and _use_native_amp:
            with autocast():
                outputs = model(**inputs)
                loss = outputs[0]
        else:
            outputs = model(**inputs)
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs

        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16 and _use_native_amp:
            scaler.scale(loss).backward()
        elif args.fp16 and _use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # unscale and clip grad norm
            if args.fp16 and _use_native_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            elif args.fp16 and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # model params step
            if args.fp16 and _use_native_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            model.zero_grad()

            if step % 10 == 0 and step > 0 and not args.debug:
                p.set_postfix(loss=round(loss.item(), 4))
                wandb.log({"epoch": epoch, "step": step, "train_loss": loss.item(),
                           "learning_rate": lr_scheduler.get_last_lr()[0]})


def eval_loop(args, model, eval_dataloader, label_map):
    model.eval()
    eval_func = None

    if args.task_name == "ner":
        from seqeval.metrics import classification_report
        y_true = []
        y_pred = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, None)
                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_map_reverse[label_ids[i][j]] == "[SEP]":
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map_reverse[label_ids[i][j]])
                        temp_2.append(label_map_reverse[logits[i][j]])

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
    elif args.task_name == "textclf":
        _all_logits = []
        _all_labels = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            inputs = {k: v.cuda() for k, v in batch.items()}
            label_ids = inputs["labels"]
            # ignore labels for inference
            inputs["labels"] = None
            logits = model(**inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            _all_logits.append(logits.detach().cpu())
            _all_labels.append(label_ids.detach().cpu())
        all_logits = torch.cat(_all_logits, 0)
        all_labels = torch.cat(_all_labels, 0)
        _, preds = torch.max(all_logits.data, 1)
        acc = np.mean((preds.byte() == all_labels.byte()).float().numpy())
        logger.info("Accuracy: " + str(acc))

    elif args.task_name == "tag":
        from evaluation.tag_eval import accuracy_with_thresh as eval_func
        from evaluation.tag_eval import roc_auc

        _all_logits = []
        _all_labels = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.cuda() for t in batch)
            if args.encode_document:
                document_batch, label_ids = batch
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
    elif args.task_name == "summary":
        from evaluation.summarization_eval import evaluate_bleu
        table = wandb.Table(columns=["Text", "Predicted Summary", "Reference Summary"])
        summary_list = []
        references_list = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            inputs = {k: v.cuda() for k, v in batch.items()}
            # FIXME we got some problems in this max length of summary in Datasets Class
            label_ids = inputs["labels"]
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=56, early_stopping=True)
            summary_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            label_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_ids]
            summary_list.extend(summary_text)
            references_list.extend(label_text)
            # get the first example from batch as wandb case
            raw_text_0 = args.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            table.add_data(raw_text_0, summary_text[0], label_text[0])
        avg_score, scores = evaluate_bleu(summary_list, references_list)
        logger.info("BLEU average score: " + str(round(avg_score, 4)))
        wandb.log({"examples": table})


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the model to do task")
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
    parser.add_argument("--adam_beta1",
                        default=0.9,
                        type=float,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2",
                        default=0.999,
                        type=float,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay")
    parser.add_argument("--warmup_steps",
                        default=200,
                        type=int,
                        help="Number of steps used for a linear warmup from 0 to `learning_rate`")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--freeze_encoder",
                        action='store_true',
                        help="Whether not to train bert encoder")
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
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level',
                        default="O1",
                        type=str,
                        help="apex fp16 optimization level")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--debug',
                        action='store_true',
                        help="in debug mode, will not enable wandb log")
    args = parser.parse_args()

    if not args.debug:
        wandb.init(project="nlp-task")

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

    task_name = args.task_name.lower()
    model_name = args.model_name.lower()
    tokenizer = get_tokenizer(args)
    args.tokenizer = tokenizer

    if model_name == "nezha":
        # copy vocab.txt from pretrained model dir to output dir
        if args.bert_model:
            shutil.copyfile(os.path.join(args.bert_model, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
        elif args.trained_model_dir and args.trained_model_dir != args.output_dir:
            shutil.copyfile(os.path.join(args.trained_model_dir, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
    # TODO: if other model name needs the vocab.txt
    # elif model_name == "OTHER"
    # pass

    if args.do_train:
        # label_map {id: label, ...}
        if args.task_name in ["ner", "textclf", "tag"]:
            if not os.path.exists(os.path.join(args.data_dir, "label_map")):
                logger.info("your task type need a label_map file under data_dir, please check!")
                exit()
            with open(os.path.join(args.data_dir, "label_map")) as f:
                label_map = json.loads(f.read().strip())
                label_map = {int(k):v for k, v in label_map.items()}
            # copy label_map to output dir
            shutil.copyfile(
                os.path.join(args.data_dir, "label_map"),
                os.path.join(args.output_dir, "label_map")
            )
        else:
            label_map = {}
        num_labels = len(label_map)

        # label_map_reverse {label: id, ...}
        label_map_reverse = {v: k for k, v in label_map.items()}

        # TODO add label_map_reverse into nerdataset
        num_examples, train_dataloader = get_dataloader(args, tokenizer, num_labels, "train")
        # TODO add control flag
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "dev")

        # total training steps (including multi epochs)
        num_training_steps = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
    else:
        # TODO not train...
        # do some other things
        pass

    if args.model_name == "nezha":
        if args.trained_model_dir:
            logger.info('init nezha model from user fine-tune model...')
            config = NeZhaConfig().from_json_file(os.path.join(args.trained_model_dir, 'bert_config.json'))
            model = get_model(args, config, num_labels=num_labels)
            model.load_state_dict(torch.load(os.path.join(args.trained_model_dir, WEIGHTS_NAME)))
        elif args.bert_model:
            logger.info('init nezha model from original pretrained model...')
            config = NeZhaConfig().from_json_file(os.path.join(args.bert_model, 'bert_config.json'))
            model = get_model(args, config, num_labels=num_labels)
            utils.torch_show_all_params(model)
            utils.torch_init_model(model, os.path.join(args.bert_model, 'pytorch_model.bin'))
    elif args.model_name == "longformer":
        logger.info('init longformer model from original pretrained model...')
        model = get_model(args, None, num_labels=num_labels)
    elif args.model_name == "bart":
        logger.info('init bart model from original pretrained model...')
        model = get_model(args, None, num_labels=num_labels)

    # check model details on wandb
    if not args.debug:
        wandb.watch(model)

    optimizer, lr_scheduler = get_optimizer_and_scheduler(args, model, num_training_steps)
    scaler = None
    model = model.cuda()
    if args.fp16 and _use_apex:
        logger.error("using apex amp for fp16...")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    elif args.fp16 and _use_native_amp:
        logger.error("using pytorch native amp for fp16...")
        scaler = torch.cuda.amp.GradScaler()
    elif args.fp16 and (_use_apex is False and _use_native_amp is False):
        logger.error("your environment DO NOT support fp16 training...")
        exit()

    if args.distributed:
        # TODO needs fix here
        model.cuda(args.local_rank)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank])
    elif num_gpus > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        logger.info("== start training on train set ==")
        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            # train loop in one epoch
            train_loop(args, model, train_dataloader, optimizer, lr_scheduler, num_gpus, epoch, scaler)

            # begin to evaluate
            logger.info("== running evaluation on dev set ==")
            eval_loop(args, model, eval_dataloader, label_map)

            # Save a trained model and the associated configuration
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())

            epoch += 1

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("== running evaluation on dev set ==")
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "dev")
        eval_loop(args, model, eval_dataloader, label_map)

    if args.do_test:
        logger.info("== running evaluation in test set ==")
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "test")
        eval_loop(args, model, eval_dataloader, label_map)


if __name__ == "__main__":
    main()
