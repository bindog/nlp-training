import os
import json
import logging
import shutil
from bidict import bidict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def get_label_map(cfg):
    # label_map {id: label, ...}
    if cfg["train"]["task_name"] in ["ner", "textclf", "tag", "pet"]:
        if not os.path.exists(os.path.join(cfg["data"]["data_dir"], "label_map")):
            logger.info("your task type need a label_map file under data_dir, please check!")
            exit()
        with open(os.path.join(cfg["data"]["data_dir"], "label_map")) as f:
            label_map = json.loads(f.read().strip())
            label_map = {int(k):v for k, v in label_map.items()}
        # copy label_map to output dir
        shutil.copyfile(
            os.path.join(cfg["data"]["data_dir"], "label_map"),
            os.path.join(cfg["train"]["output_dir"], "label_map")
        )
        num_labels = len(label_map)
    else:
        label_map = {}
        num_labels = -1

    label_map = bidict(label_map)
    return label_map, num_labels


def get_split_path(data_dir, split):
    if split == "val" or split == "dev" or split == "valid":
        c_list = ["val", "dev", "valid"]
        for c in c_list:
            json_path = os.path.join(data_dir, c + ".json")
            if os.path.exists(json_path):
                return json_path
    return os.path.join(data_dir, split + ".json")


def parse_pet_config(cfg):
    if "pet" in cfg:
        pet_pattern = {
            "position": cfg["pet"]["pet_pattern"]["position"],
            "placeholder": cfg["pet"]["pet_pattern"]["placeholder"],
            "prefix": cfg["pet"]["pet_pattern"]["prefix"],
            "suffix": cfg["pet"]["pet_pattern"]["suffix"]
        }
        cwords_list = cfg["pet"]["candidate_words"]
        candidate_words = {i: cwords_list[i] for i in range(len(cwords_list))}
        return pet_pattern, candidate_words
    else:
        logger.info("pet config not found in your yaml config")
        exit()


def get_dataloader(cfg, tokenizer, num_labels, split, debug=False):
    # speed up in debug mode
    if debug:
        split = "dev"

    json_file = get_split_path(cfg["data"]["data_dir"], split)
    if cfg["train"]["task_name"] == "ner":
        from datasets.ner import NERDataset
        label_map_path = os.path.join(cfg["data"]["data_dir"], "label_map")
        dataset = NERDataset(json_file, label_map_path, tokenizer, num_labels=num_labels)
    elif cfg["train"]["task_name"] == "textclf":
        longformer = False
        from datasets.textclf import TextclfDataset
        dataset = TextclfDataset(json_file, tokenizer, num_labels, cfg["train"]["doc_inner_batch_size"], cfg["data"]["max_seq_length"], cfg["train"]["encode_document"], longformer)
    elif cfg["train"]["task_name"] == "tag":
        longformer = False
        from datasets.textclf import TextclfDataset
        dataset = TextclfDataset(json_file, tokenizer, num_labels, cfg["train"]["doc_inner_batch_size"], cfg["data"]["max_seq_length"], cfg["train"]["encode_document"], longformer, tag=True)
    elif cfg["train"]["task_name"] == "summary":
        from datasets.summarization import SummarizationDataset
        dataset = SummarizationDataset(json_file, tokenizer, max_source_length=cfg["data"]["max_src_length"], max_target_length=cfg["data"]["max_tgt_length"])
    elif cfg["train"]["task_name"] == "translation":
        from datasets.translation import TranslationDataset
        dataset = TranslationDataset(json_file, tokenizer)
    elif cfg["train"]["task_name"] == "pet":
        from datasets.pet import PetDataset
        label_map, num_labels = get_label_map(cfg)
        cfg["data"]["num_labels"] = num_labels
        pet_pattern, candidate_words = parse_pet_config(cfg)
        # TODO temp setting for google mt5
        cfg["data"]["max_tgt_length"] = 3 + len(candidate_words[0])
        dataset = PetDataset(json_file, tokenizer, pet_pattern, candidate_words, label_map, max_length=cfg["data"]["max_src_length"])

    if cfg["system"]["distributed"]:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    shuffle = True if split == "train" and cfg["system"]["distributed"] == False else False
    batch_size = cfg["train"]["batch_size"] if split == "train" else cfg["eval"]["batch_size"]
    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=4,
                        sampler=sampler
                    )
    return len(dataset), dataloader
