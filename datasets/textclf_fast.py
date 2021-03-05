import os
import logging
import torch

from lmdb_embeddings.reader import LruCachedLmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError

from .lmdb_dataset import LMDBDataset


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


class TextclfDatasetFast(LMDBDataset):
    def __init__(self, lmdb_path, lmdb_embeddings_path=None, tokenizer=None, num_labels=None, max_length=None):
        """Initiate Textclf Dataset dataset.
        Arguments:
            json_path: dataset json file
            tokenizer: BERT tokenizer
            num_labels: number of labels
            max_seq_per_doc: enabled if encode_document is True, the max inner batch_size of a document
            max_seq_length: the max sequence length which BERT supports
            encode_document: whether treat the text as a document
            longformer: as a document, but use the longformer model and longformer tokenizer
            tag: the labels will be multi_label if True, else just a normal label
        """
        super().__init__(lmdb_path)

        assert lmdb_embeddings_path or tokenizer, "At least one of lmdb_embeddings_path or tokenizer should be given"

        if lmdb_embeddings_path is not None:
            self.embeddings = LruCachedLmdbEmbeddingsReader(lmdb_embeddings_path)
        if tokenizer is not None:
            self.tokenizer = tokenizer

    def process(self, raw_dict):
        valid_words = raw_dict["words"]
        all_word_tensors = []
        for word in valid_words:
            try:
                vector = self.embeddings.get_word_vector(word)
                word_tensor = torch.tensor(vector, dtype=torch.float32)
                all_word_tensors.append(word_tensor)
            except MissingWordError:
                pass
        # reduce mean all vectors
        return {
            "input_embeddings": torch.mean(torch.stack(all_word_tensors), axis=0),
            "labels": raw_dict["category"]
        }
