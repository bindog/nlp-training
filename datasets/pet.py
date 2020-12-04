import os
import json
import torch
import logging
from bidict import bidict


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


class PetDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, pet_pattern, candidate_words, label_map, max_length=1024):
        """Initiate sentiment dataset with pet training pattern.
        Arguments:
            json_path: dataset json file
            tokenizer: tokenizer
            pet_pattern: the pattern of pet training mode, an example:
                        pet_pattern = {
                                        "position": "head",
                                        "placeholder": "<extra_id_0>",
                                        "prefix": "这是一篇",
                                        "suffix": "报道。"
                                    }
                        position == "head" means we put the pet pattern in the beginning of the src text,
                        placeholder means the position of the cloze words, the prefix and suffix means the
                        text around the placeholder
            candidate_words: the candiate words to fill in the placeholder, an example:
                        candidate_words = bidict({
                            0: "政治",
                            1: "军事",
                            2: "经济",
                            3: "科技",
                        })
            label_map: the raw id -> label mapping, please use bidict
            max_length: the max src text length
        """
        super(PetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # TODO temp setting for google mt5
        # all candidate_words should have the same length
        self.max_target_length = 3 + len(candidate_words[0])
        self.pet_pattern = pet_pattern
        self.candidate_words = candidate_words
        self.label_map = label_map

        self.all_batch = []
        self.all_raw_label = []

        # NOTE: pet training mode supports a few shot learning,
        # means we have little dataset to train, if you have large
        # dataset, please consider using normal training tasks

        self.total_num = 0
        with open(json_path, "r") as f:
            for line in f:
                line_dict = json.loads(line.strip())
                text = line_dict["text"]
                label = self.label_map.inv[line_dict["label"]]

                batch = tokenizer.prepare_seq2seq_batch(
                    src_texts=[self.pet_pattern["prefix"] + \
                               " " + self.pet_pattern["placeholder"] + " " + \
                               self.pet_pattern["suffix"] + text],
                    tgt_texts=[self.pet_pattern["placeholder"] + " " + \
                               self.candidate_words[label]],
                    max_length=self.max_length,
                    max_target_length=self.max_target_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                _batch = {k: v.squeeze() for k, v in batch.items()}
                self.all_batch.append(_batch)
                self.all_raw_label.append(label)
                self.total_num += 1

    def __getitem__(self, i):
        return_dict = {}
        return_dict.update(self.all_batch[i])
        # return_dict["raw_label"] = self.all_raw_label[i]
        return return_dict

    def __len__(self):
        return self.total_num
