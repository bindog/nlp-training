import os
import shutil
import torch
import pathlib
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


pretrained_local_mapping = {
    "huawei/nezha-en-base": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-en-base",
    "huawei/nezha-zh-base": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-zh-base",
    "huawei/nezha-zh-large": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-zh-large",
    "facebook/mbart-large-cc25": "/mnt/dl/public/pretrained_models/mbart-large-cc25",
    "google/mt5-base": "/mnt/dl/public/pretrained_models/mt5-base",
    "google/mt5-large": "/mnt/dl/public/pretrained_models/mt5-large",
    "hfl/chinese-bert-wwm": "/mnt/dl/public/pretrained_models/chinese-bert-wwm",
    "hfl/chinese-bert-wwm-ext": "/mnt/dl/public/pretrained_models/chinese-bert-wwm-ext",
    "hfl/chinese-roberta-wwm-ext": "/mnt/dl/public/pretrained_models/chinese-roberta-wwm-ext",
    "hfl/chinese-roberta-wwm-ext-large": "/mnt/dl/public/pretrained_models/chinese-roberta-wwm-ext-large"
}

nlu_tasks = ["ner", "textclf", "tag", "sentiment"]
nlg_tasks = ["summary", "translation", "pet"]


def get_tokenizer(cfg):
    if cfg["train"]["model_name"] == "nezha":
        from models.tokenization_bert import BertTokenizer
        if cfg["train"]["pretrained_model"]:
            tokenizer = BertTokenizer.from_pretrained(cfg["train"]["pretrained_model"])
        else:
            logger.error("BERT vocab file not set, please check your ber_model_dir or trained_model_dir")
        logger.info('vocab size is %d' % (len(tokenizer.vocab)))
        return tokenizer
    elif cfg["train"]["model_name"] == "bart" or cfg["train"]["model_name"] == "mbart":
        from models.tokenization_mbart import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
        return tokenizer
    elif cfg["train"]["model_name"] == "t5" or cfg["train"]["model_name"] == "mt5":
        from models.tokenization_t5 import T5Tokenizer
        pretrained_tag = "/mnt/dl/public/pretrained_models/mt5-large"
        tokenizer = T5Tokenizer.from_pretrained(pretrained_tag)
        return tokenizer
    elif cfg["train"]["model_name"] == "bert" or cfg["train"]["model_name"] == "hfl":
        from models.tokenization_bert import BertTokenizer
        pretrained_tag = "/mnt/bigfiles/Models/pretrained_models/chinese_roberta_wwm_ext_pytorch"
        tokenizer = BertTokenizer.from_pretrained(pretrained_tag)
        return tokenizer
    else:
        logger.error("can not find the proper tokenizer type...")
        return None


def get_pretrained_model_path(cfg):
    ptd = None
    # pretrained_model path comes first, cause it maybe user specified trained model
    if cfg["train"]["pretrained_model"]:
        if pathlib.Path(cfg["train"]["pretrained_model"]).exists():
            ptd = cfg["train"]["pretrained_model"]
            return ptd

    # then we check the pretrained_tag same to HuggingFace
    if cfg["train"]["pretrained_tag"] in pretrained_local_mapping:
        # default path
        ptd = pretrained_local_mapping[cfg["train"]["pretrained_tag"]]
    else:
        ptd = cfg["train"]["pretrained_model"]
    assert pathlib.Path(ptd).exists(), "pretrained model path not exists, please save it to public folder first!"
    return ptd


def get_tokenizer_and_model(cfg, label_map=None, num_labels=None):
    if num_labels is None:
        num_labels = cfg["data"]["num_labels"]

    tokenizer = None
    model = None
    ptd = get_pretrained_model_path(cfg)

    # Huawei nezha
    if cfg["train"]["model_name"] == "nezha":
        from models.tokenization_bert import BertTokenizer
        from models.modeling_nezha import (
                            NeZhaForSequenceClassification, NeZhaForTokenClassification,
                            NeZhaBiLSTMForTokenClassification, NeZhaForDocumentClassification,
                            NeZhaForDocumentTagClassification, NeZhaForTagClassification
                        )
        tokenizer = BertTokenizer.from_pretrained(ptd)

        if cfg["train"]["task_name"] == "ner":
            if cfg["train"]["use_bilstm"]:
                _label_map = {k:v for k,v in label_map.items()}
                model = NeZhaBiLSTMForTokenClassification.from_pretrained(ptd, label_map=_label_map, num_labels=num_labels)
            else:
                model = NeZhaForTokenClassification.from_pretrained(ptd, num_labels=num_labels)
        if cfg["train"]["task_name"] == "textclf":
            if cfg["train"]["encode_document"]:
                model = NeZhaForDocumentClassification.from_pretrained(ptd, doc_inner_batch_size=cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
            else:
                model = NeZhaForSequenceClassification.from_pretrained(ptd, num_labels=num_labels)
        if cfg["train"]["task_name"] == "tag":
            if cfg["train"]["encode_document"]:
                model = NeZhaForDocumentTagClassification.from_pretrained(ptd, doc_inner_batch_size=cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
            else:
                model = NeZhaForTagClassification.from_pretrained(ptd, num_labels=num_labels)

    # facebook bert and XunFei hfl
    elif cfg["train"]["model_name"] == "bert" or cfg["train"]["model_name"] == "hfl":
        from models.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(ptd)
        if cfg["train"]["task_name"] == "ner":
            from models.modeling_bert import BertForTokenClassification
            if cfg["train"]["use_bilstm"]:
                # FIXME Process BiLSTM
                # model = NeZhaBiLSTMForTokenClassification(bert_config, label_map, num_labels=num_labels)
                pass
            else:
                model = BertForTokenClassification.from_pretrained(ptd, num_labels=num_labels)
        if cfg["train"]["task_name"] == "textclf":
            from models.modeling_bert import BertForSequenceClassification
            if cfg["train"]["encode_document"]:
                # FIXME Process NeZhaForDocumentClassification
                # model = NeZhaForDocumentClassification(bert_config, cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
                pass
            else:
                model = BertForSequenceClassification.from_pretrained(ptd, num_labels=num_labels)
        if cfg["train"]["task_name"] == "tag":
            from models.modeling_bert import BertForTagClassification
            if cfg["train"]["encode_document"]:
                # FIXME Process NeZhaForDocumentTagClassification
                # model = NeZhaForDocumentTagClassification(bert_config, cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
                pass
            else:
                model = BertForTagClassification.from_pretrained(ptd, num_labels=num_labels)

    # facebook bart/mbart
    elif cfg["train"]["model_name"] == "bart" or cfg["train"]["model_name"] == "mbart":
        from models.tokenization_mbart import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained(ptd)
        if cfg["train"]["task_name"] in nlg_tasks:
            from models.modeling_mbart import MBartForConditionalGeneration
            gradient_checkpointing_flag = True if cfg["train"]["gradient_checkpointing"] else False
            if gradient_checkpointing_flag:
                logger.info("gradient checkpointing enabled")
            model = MBartForConditionalGeneration.from_pretrained(ptd, gradient_checkpointing=gradient_checkpointing_flag)

    # google t5/mt5
    elif cfg["train"]["model_name"] == "t5" or cfg["train"]["model_name"] == "mt5":
        from models.tokenization_t5 import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(ptd)
        if cfg["train"]["task_name"] in nlg_tasks:
            from models.modeling_mt5 import MT5ForConditionalGeneration
            model = MT5ForConditionalGeneration.from_pretrained(ptd)
    elif cfg["train"]["model_name"] == "simple":
        from models_classic.simple_net import SimpleFC
        tokenizer = "word2vec"
        model = SimpleFC(embedding_dim=200, num_labels=num_labels)
    else:
        logger.error("model type not supported!")

    assert tokenizer and model, "get tokenizer or model error"
    if cfg["train"]["freeze_encoder"] and "freeze_encoder" in dir(model):
        model.freeze_encoder()
        if "unfreeze_encoder_last_layers" in dir(model):
            model.unfreeze_encoder_last_layers()
    return tokenizer, model


def save_model(cfg, tokenizer, model, best=False, epoch=None, step=None):
    if best:
        sub_dir = "best_models"
    else:
        if epoch is not None:
            assert isinstance(epoch, int) and isinstance(step, int), "type of epoch and step should be int"
            sub_dir = "models_ep_" + str(epoch) + "_step_" + str(step)
        else:
            sub_dir = "models"

    saved_path = os.path.join(cfg["train"]["output_dir"], sub_dir)
    os.makedirs(saved_path, exist_ok=True)

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(saved_path)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(saved_path)
    else:
        _model = model.module if hasattr(model, 'module') else model  # handle multi gpu
        state_dict = _model.state_dict()
        output_model_file = os.path.join(saved_path, "pytorch_model.bin")
        torch.save(state_dict, output_model_file)
