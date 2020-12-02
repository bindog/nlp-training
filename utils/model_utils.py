import os
import shutil
import torch
import pathlib
import logging
from tools import official_tokenization as tokenization, utils


pretrained_local_mapping = {
    "huawei/nezha-en-base": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-en-base",
    "huawei/nezha-zh-base": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-zh-base",
    "huawei/nezha-zh-large": "/mnt/dl/public/pretrained_models/NEZHA-Models/nezha-zh-large",
    "facebook/mbart-large-cc25": "/mnt/dl/public/pretrained_models/mbart-large-cc25",
    "google/mt5-base": "/mnt/dl/public/pretrained_models/mt5-base",
    "google/mt5-large": "/mnt/dl/public/pretrained_models/mt5-large",
    "hfl/chinese-roberta-wwm-ext": "/mnt/dl/public/pretrained_models/chinese_roberta_wwm_ext/",
    "hfl/chinese-roberta-wwm-ext-large/": "/mnt/dl/public/pretrained_models/chinese-roberta-wwm-ext-large/"
}


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def get_tokenizer(cfg):
    if cfg["train"]["model_name"] == "nezha":
        if cfg["train"]["pretrained_model"]:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(cfg["train"]["pretrained_model"], "vocab.txt"), do_lower_case=True)
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
        from models.modeling_nezha import (
                            NeZhaConfig, NeZhaForSequenceClassification, NeZhaForTokenClassification,
                            NeZhaBiLSTMForTokenClassification, NeZhaForDocumentClassification,
                            NeZhaForDocumentTagClassification, NeZhaForTagClassification
                        )
        bert_config = NeZhaConfig().from_json_file(os.path.join(ptd, "bert_config.json"))
        if ptd:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(ptd, "vocab.txt"), do_lower_case=True)
            logger.info('vocab size is %d' % (len(tokenizer.vocab)))
        else:
            logger.error("BERT vocab file not set, please check your ber_model_dir or trained_model_dir")

        if cfg["train"]["task_name"] == "ner":
            label_map_reverse = {v: k for k, v in label_map.items()}
            if cfg["train"]["ner_addBilstm"]:
                model = NeZhaBiLSTMForTokenClassification(bert_config, label_map_reverse, num_labels=num_labels)
            else:
                model = NeZhaForTokenClassification(bert_config, num_labels=num_labels)
        if cfg["train"]["task_name"] == "textclf":
            if cfg["train"]["encode_document"]:
                model = NeZhaForDocumentClassification(bert_config, cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
            else:
                model = NeZhaForSequenceClassification(bert_config, num_labels=num_labels)
        if cfg["train"]["task_name"] == "tag":
            if cfg["train"]["encode_document"]:
                model = NeZhaForDocumentTagClassification(bert_config, cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
            else:
                model = NeZhaForTagClassification(bert_config, num_labels=num_labels)
        # FIXME pytorch_model.bin path subdir
        if cfg["train"]["pretrained_model"] == "":
            utils.torch_init_model(model, os.path.join(ptd, 'pytorch_model.bin'))
        elif pathlib.Path(cfg["train"]["pretrained_model"]).exists():
            model.load_state_dict(torch.load(os.path.join(ptd, 'pytorch_model.bin')))
        else:
            raise('pretrained_model Path error!')

    # facebook bert and XunFei hfl
    elif cfg["train"]["model_name"] == "bert" or cfg["train"]["model_name"] == "hfl":
        from models.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(pretrained_local_mapping[cfg["train"]["pretrained_tag"]])
        if cfg["train"]["task_name"] == "ner":
            from models.modeling_bert import BertForTokenClassification
            label_map_reverse = {v: k for k, v in label_map.items()}
            if cfg["train"]["ner_addBilstm"]:
                # FIXME Process BiLSTM
                # model = NeZhaBiLSTMForTokenClassification(bert_config, label_map_reverse, num_labels=num_labels)
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
            if cfg["train"]["encode_document"]:
                # FIXME Process NeZhaForDocumentTagClassification                
                # model = NeZhaForDocumentTagClassification(bert_config, cfg["train"]["doc_inner_batch_size"], num_labels=num_labels)
                pass
            else:
                # FIXME Process NeZhaForTagClassification    
                # model = NeZhaForTagClassification(bert_config, num_labels=num_labels)
                pass

    # facebook bart/mbart
    elif cfg["train"]["model_name"] == "bart" or cfg["train"]["model_name"] == "mbart":
        from models.tokenization_mbart import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained(pretrained_local_mapping[cfg["train"]["pretrained_tag"]])
        if cfg["train"]["task_name"] == "summary" or cfg["train"]["task_name"] == "translation":
            from models.modeling_mbart import MBartForConditionalGeneration
            gradient_checkpointing_flag = True if cfg["train"]["gradient_checkpointing"] else False
            if gradient_checkpointing_flag:
                logger.info("gradient checkpointing enabled")
            model = MBartForConditionalGeneration.from_pretrained(ptd, gradient_checkpointing=gradient_checkpointing_flag)
    
    # google t5/mt5
    elif cfg["train"]["model_name"] == "t5" or cfg["train"]["model_name"] == "mt5":
        from models.tokenization_t5 import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(ptd)
        if cfg["train"]["task_name"] == "summary" or cfg["train"]["task_name"] == "translation":
            from models.modeling_mt5 import MT5ForConditionalGeneration
            model = MT5ForConditionalGeneration.from_pretrained(ptd)
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

    if cfg["train"]["model_name"] == "nezha":
        # Save vocab.txt
        ptd = get_pretrained_model_path(cfg)
        shutil.copyfile(os.path.join(ptd, "vocab.txt"), os.path.join(cfg["train"]["output_dir"], "vocab.txt"))
        # Save a trained model and the associated configuration
        if not pathlib.Path(os.path.join(cfg["train"]["output_dir"], sub_dir)).exists():
            os.makedirs(os.path.join(cfg["train"]["output_dir"], sub_dir))
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(cfg["train"]["output_dir"], sub_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(cfg["train"]["output_dir"], sub_dir, "bert_config.json")
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
    else:
        saved_path = os.path.join(cfg["train"]["output_dir"], sub_dir)
        tokenizer.save_pretrained(saved_path)
        model.save_pretrained(saved_path)
