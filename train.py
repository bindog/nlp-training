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
from evaluation import eval_wrapper

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
        if debug:
            break


def eval_loop(cfg, tokenizer, model, eval_dataloader, debug=False):
    model.eval()
    pred_list = []
    label_list = []
    table = wandb.Table(columns=["Text", "Predicted", "GroundTruth"])

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        label_ids = batch.pop("labels")
        inputs = {k: v.cuda() for k, v in batch.items()}

        # natural language understanding, directly inference
        if cfg["eval"]["type"] == "nlu":
            with torch.no_grad():
                logits = model(**inputs)
                if isinstance(logits, tuple):
                    logits = logits[0]
            pred_list.append(logits.detach().cpu())
            label_list.append(label_ids)
            # FIXME get the predicted label and ground truth label
            raw_text_0 = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            table.add_data(raw_text_0, pred_list[0], label_list[0])
        # natural language generation, using generate and beam search
        elif cfg["eval"]["type"] == "nlg":
            input_ids = inputs["input_ids"].cuda()
            pred_ids = model.generate(
                                    input_ids,
                                    num_beams=cfg["eval"]["num_beams"],
                                    max_length=cfg["data"]["max_tgt_length"],
                                    early_stopping=cfg["eval"]["early_stopping"]
                                )
            pred_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in pred_ids]
            label_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in label_ids]
            pred_list.extend(pred_text)
            label_list.extend(label_text)

            # get the first example from the batch as wandb demo case
            raw_text_0 = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            table.add_data(raw_text_0, pred_text[0], label_text[0])

        if debug:
            break

    results = eval_wrapper(cfg, pred_list, label_list)
    logger.info("Result of evaluation metric: " + cfg["eval"]["metric"])
    for k, v in results.items():
        logger.info(k + ": " + str(v))
    if not debug:
        wandb.log(results)

    key_imp = list(results.keys())[0]
    return results[key_imp]


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
        wandb.run.name = cfg["data"]["corpus"] + '-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        wandb.config.update(args)
        wandb.run.save()

    if args.local_rank == -1:
        num_gpus = torch.cuda.device_count()
        args.distributed = False
        # TODO multi gpus
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

    num_examples, train_dataloader = get_dataloader(cfg, tokenizer, num_labels, "train", debug=args.debug)
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
        # TODO add devices id here
        model = torch.nn.DataParallel(model)

    # Train
    logger.info("start training on train set")
    epoch = 0
    best_score = -1
    for _ in trange(int(cfg["train"]["train_epochs"]), desc="Epoch"):
        best = False
        # train loop in one epoch
        train_loop(cfg, model, train_dataloader, optimizer, lr_scheduler, num_gpus, epoch, scaler, args.debug)
        # begin to evaluate
        logger.info("running evaluation on dev set")
        score = eval_loop(cfg, tokenizer, model, eval_dataloader, args.debug)
        if best_score < score:
            best_score = score
            best = True
        # Save a trained model and the associated configuration
        save_model(cfg, tokenizer, model, best)

        epoch += 1

    # Test Eval
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("running evaluation on final test set")
        # TODO new test set?
        _, eval_dataloader = get_dataloader(cfg, tokenizer, num_labels, "dev", debug=args.debug)
        score = eval_loop(cfg, tokenizer, model, eval_dataloader, args.debug)


if __name__ == "__main__":
    main()
