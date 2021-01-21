import sys
sys.path.insert(0, "..")

import os
import json
import logging
import argparse
import sentencepiece as spm

from models.tokenization_t5 import T5Tokenizer
from models.modeling_mt5 import MT5ForConditionalGeneration
from models.tokenization_mbart import MBartTokenizer
from models.modeling_mbart import MBartForConditionalGeneration


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="reducing transformer tokenizer size")
    parser.add_argument("--source_model",
                        type=str,
                        required=True,
                        default='',
                        help="The multilingual transformer to start from")
    parser.add_argument("--custom_corpus",
                        type=str,
                        required=True,
                        default='custom_corpus.txt',
                        help="the custom corpus similar to target corpus with limited tokens")
    parser.add_argument("--vocab_size",
                        type=int,
                        required=True,
                        default=8000,
                        help="vocabulary size")
    parser.add_argument("--output_model",
                        type=str,
                        required=True,
                        default='output_model',
                        help="The name of the final reduced model")
    args = parser.parse_args()

    # Load original tokenizer, model and vocab
    logging.info('starting from model: ' + args.source_model)
    if "mt5" in args.source_model:
        tokenizer = T5Tokenizer.from_pretrained(args.source_model)
        model_type = "mt5"
    elif "mbart" in args.source_model:
        tokenizer = MBartTokenizer.from_pretrained(args.source_model)
        model_type = "mbart"
    else:
        logging.info("model type not supported...")
        exit()
    vocab = tokenizer.get_vocab()

    spm.SentencePieceTrainer.train(
        input=args.custom_corpus,
        model_prefix=os.path.join(args.output_model, "reduce_sentencepiece.bpe"),
        vocab_size=args.vocab_size,
        model_type="bpe",
        vocabulary_output_piece_score=False
    )

    bpe_model_path = os.path.join(args.output_model, "reduce_sentencepiece.bpe.model")
    if model_type == "mt5":
        new_tokenizer = T5Tokenizer(vocab_file=bpe_model_path)
        new_tokenizer.save_pretrained(args.output_model)
    elif model_type == "mbart":
        new_tokenizer = MBartTokenizer(vocab_file=bpe_model_path)
        new_tokenizer.save_pretrained(args.output_model)
    else:
        logging.info("model type not supported...")
        exit()

if __name__ == "__main__":
    main()
