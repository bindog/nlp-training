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
import pathlib
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
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm, trange

from utils.config_utils import parse_cfg
from utils.data_utils import get_label_map, get_dataloader
from utils.model_utils import get_tokenizer_and_model, save_model
from utils.optimization import AdamW, get_linear_schedule_with_warmup

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


def train_loop(cfg, model, train_dataloader, optimizer, lr_scheduler, num_gpus, epoch, scaler=None, debug=False):
    model.train()
    p = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(p):
        inputs = {k: v.cuda() for k, v in batch.items()}

        if cfg["train"]["fp16"] and _use_native_amp:
            with autocast():
                outputs = model(**inputs)
                loss = outputs[0]
        # FIXME ner bilstm condition not correct
        # elif cfg["train"]["ner_addBilstm"]:
        #     loss = model.neg_log_likelihood(**inputs)
        else:
            outputs = model(**inputs)
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss

        if num_gpus > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if cfg["optimizer"]["gradient_accumulation_steps"] > 1:
            loss = loss / cfg["optimizer"]["gradient_accumulation_steps"]

        if cfg["train"]["fp16"] and _use_native_amp:
            scaler.scale(loss).backward()
        elif cfg["train"]["fp16"] and _use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (step + 1) % cfg["optimizer"]["gradient_accumulation_steps"] == 0:
            # unscale and clip grad norm
            if cfg["train"]["fp16"] and _use_native_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["max_grad_norm"])
            elif cfg["train"]["fp16"] and _use_apex:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg["optimizer"]["max_grad_norm"])
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["max_grad_norm"])

            # model params step
            if cfg["train"]["fp16"] and _use_native_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            lr_scheduler.step()
            # model.zero_grad()
            optimizer.zero_grad()

            if step % 10 == 0 and step > 0 and not debug:
                p.set_postfix(loss=round(loss.item(), 4))
                wandb.log({"epoch": epoch, "step": step, "train_loss": loss.item(),
                           "lr": lr_scheduler.get_last_lr()[0]})
        # if debug:
        #     break


def eval_loop(args, model, eval_dataloader, label_map):
    model.eval()
    eval_func = None

    if args.task_name == "ner":
        from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
        y_true = []
        y_pred = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            inputs = {k: v.cuda() for k, v in batch.items()}
            label_ids = inputs["labels"]
            inputs["labels"] = None

            with torch.no_grad():
                if args.ner_addBilstm:
                    logits = model(**inputs)
                else:
                    logits = model(**inputs)
                    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_map[label_ids[i][j]] == "[SEP]":
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

            if args.debug:
                break
        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)
        eval_precision = precision_score(y_true, y_pred)
        eval_recall = recall_score(y_true, y_pred)
        eval_f1 = f1_score(y_true, y_pred)
        if not args.debug:
            wandb.log({"eval_precision": eval_precision, "eval_recall": eval_recall, "eval_f1": eval_f1})

    elif args.task_name == "textclf":
        _all_logits = []
        _all_labels = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            inputs = {k: v.cuda() for k, v in batch.items()}
            label_ids = inputs["labels"]
            # ignore labels for inference
            inputs["labels"] = None
            with torch.no_grad():
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
                # TODO fix
                document_batch, label_ids = batch
                with torch.no_grad():
                    logits = model(document_batch, None)
            else:
                # TODO fix
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.no_grad():
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
            # FIXME do we need with torch.no_grad() here?
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=args.max_summarization_length, early_stopping=True)
            summary_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            label_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_ids]
            summary_list.extend(summary_text)
            references_list.extend(label_text)
            # get the first example from batch as wandb case
            raw_text_0 = args.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            table.add_data(raw_text_0, summary_text[0], label_text[0])
        avg_score, scores = evaluate_bleu(summary_list, references_list)
        logger.info("BLEU average score: " + str(round(avg_score, 4)))
        if not args.debug:
            wandb.log({"examples": table})
    elif args.task_name == "translation":
        from evaluation.summarization_eval import evaluate_bleu
        table = wandb.Table(columns=["Text", "Predicted translation", "Reference translation"])
        translation_list = []
        references_list = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            inputs = {k: v.cuda() for k, v in batch.items()}
            # FIXME we got some problems in this max length of summary in Datasets Class
            label_ids = inputs["labels"]
            # FIXME do we need with torch.no_grad() here?
            translation_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=56, early_stopping=True)
            translation_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in translation_ids]
            label_text = [args.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_ids]
            translation_list.extend(translation_text)
            references_list.extend(label_text)
            # get the first example from batch as wandb case
            raw_text_0 = args.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            table.add_data(raw_text_0, translation_text[0], label_text[0])
        avg_score, scores = evaluate_bleu(translation_list, references_list)
        logger.info("BLEU average score: " + str(round(avg_score, 4)))
        if not args.debug:
            wandb.log({"examples": table, "score_BLEU": avg_score})


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        required=True,
                        help="the training config file")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--multi_task",
                        action="store_true",
                        help="training with multi task schema")
    parser.add_argument("--debug",
                        action="store_true",
                        help="in debug mode, will not enable wandb log")
    args = parser.parse_args()
    cfg = parse_cfg(pathlib.Path(args.config))

    cfg["train"]["output_dir"] = cfg["train"]["output_dir"] + "/" + \
                                 cfg["train"]["task_name"] + "_" + \
                                 cfg["train"]["model_name"] + "_" + \
                                 cfg["data"]["corpus"]

    output_dir_pl = pathlib.Path(cfg["train"]["output_dir"])
    if output_dir_pl.exists():
        logger.warn("output directory ({}) already exists!".format(output_dir_pl))
        time.sleep(2)
    else:
        output_dir_pl.mkdir(parents=True, exist_ok=True)

    if not args.debug:
        wandb.init(project="nlp-task", dir=cfg["train"]["output_dir"])
        wandb.run.name = args.corpus + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        wandb.config.update(args)
        wandb.run.save()

    if args.local_rank == -1:
        num_gpus = torch.cuda.device_count()
        args.distributed = False
        # FIXME multi gpus
        torch.cuda.set_device(int(cfg["system"]["cuda_devices"]))
    else:
        torch.cuda.set_device(args.local_rank)
        num_gpus = 1
        args.distributed = True
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    logger.info("num_gpus: {}, distributed training: {}, 16-bits training: {}".format(
        num_gpus, bool(args.local_rank != -1), cfg["train"]["fp16"]))
    cudnn.benchmark = True

    if cfg["optimizer"]["gradient_accumulation_steps"] < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(cfg["optimizer"]["gradient_accumulation_steps"]))

    # true batch_size in training
    cfg["train"]["batch_size"] = cfg["train"]["batch_size"] // cfg["optimizer"]["gradient_accumulation_steps"]

    # the type of label_map is bidict
    # label_map[x] = xx, label_map.inv[xx] = x
    label_map, num_labels = get_label_map(cfg)
    tokenizer, model = get_tokenizer_and_model(cfg, label_map, num_labels)

    # check model details on wandb
    if not args.debug:
        wandb.watch(model)

    # TODO add label_map_reverse into nerdataset
    num_examples, train_dataloader = get_dataloader(cfg, tokenizer, num_labels, "train", debug=args.debug)
    # TODO add control flag
    _, eval_dataloader = get_dataloader(cfg, tokenizer, num_labels, "dev", debug=args.debug)

    # total training steps (including multi epochs)
    num_training_steps = int(len(train_dataloader) // cfg["optimizer"]["gradient_accumulation_steps"] * cfg["train"]["train_epochs"])

    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg["optimizer"]["lr"]
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["optimizer"]["num_warmup_steps"], num_training_steps=num_training_steps
    )

    scaler = None
    model = model.cuda()
    if cfg["train"]["fp16"] and _use_apex:
        logger.error("using apex amp for fp16...")
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    elif cfg["train"]["fp16"] and _use_native_amp:
        logger.error("using pytorch native amp for fp16...")
        scaler = torch.cuda.amp.GradScaler()
    elif cfg["train"]["fp16"] and (_use_apex is False and _use_native_amp is False):
        logger.error("your environment DO NOT support fp16 training...")
        exit()

    if cfg["system"]["distributed"]:
        # TODO needs fix here
        model.cuda(args.local_rank)
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank])
    elif num_gpus > 1:
        # TODO add devices id here!!!
        model = torch.nn.DataParallel(model)

    # Train
    logger.info("start training on train set")
    epoch = 0
    for _ in trange(int(cfg["train"]["train_epochs"]), desc="Epoch"):
        # train loop in one epoch
        train_loop(cfg, model, train_dataloader, optimizer, lr_scheduler, num_gpus, epoch, scaler, args.debug)

        # begin to evaluate
        logger.info("running evaluation on dev set")
        eval_loop(args, model, eval_dataloader, label_map)

        # Save a trained model and the associated configuration
        save_model(cfg, tokenizer, model)

        epoch += 1

    # Eval
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("running evaluation on dev set")
        _, eval_dataloader = get_dataloader(args, tokenizer, num_labels, "dev")
        eval_loop(args, model, eval_dataloader, label_map)


if __name__ == "__main__":
    main()
