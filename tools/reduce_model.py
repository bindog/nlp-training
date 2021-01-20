import sys
sys.path.insert(0, "..")

import os
import json
import logging
import argparse
from torch import nn

from models.tokenization_t5 import T5Tokenizer
from models.modeling_mt5 import MT5ForConditionalGeneration
from models.tokenization_mbart import MBartTokenizer
from models.modeling_mbart import MBartForConditionalGeneration


logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


def select_embeddings(model, old_vocab, new_vocab, model_type="mt5", model_name='new_model'):
    # Get old embeddings from model
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

    if model_type == "mt5":
        if old_num_tokens != len(old_vocab) + 12:  # NOTE: unknow bugs, why mt5 has 12 more embeddings than vocab size
            logging.info('len(old_vocab) != len(model.old_embeddings)')
            return old_embeddings
    else:
        if old_num_tokens != len(old_vocab):
            logging.info('len(old_vocab) != len(model.old_embeddings)')
            return old_embeddings

    new_num_tokens = len(new_vocab)
    if new_vocab is None:
        logging.info('nothing to copy')
        return old_embeddings

    # Build new embeddings
    logging.info('reducing model size ...')
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device)

    # Copy weights input embeddings
    logging.info("reducing input embeddings ...")
    i = 0
    j = 0
    vocab = []
    for token in old_vocab:
        if token in new_vocab:
            vocab.append(token)
            new_embeddings.weight.data[i, :] = old_embeddings.weight.data[j, :]
            i += 1
        j += 1

    model.set_input_embeddings(new_embeddings)

    # Copy weights output embeddings if exists
    old_lm_head = model.get_output_embeddings()
    if old_lm_head is not None:
        if old_lm_head.out_features > new_num_tokens:
            logging.info("reducing output embeddings ...")
            new_lm_head = nn.Linear(model.config.d_model, new_num_tokens, bias=False)
            new_lm_head.to(old_lm_head.weight.device)
            # Copy weights output embeddings
            i = 0
            j = 0
            for token in old_vocab:
                if token in new_vocab:
                    new_lm_head.weight.data[i, :] = old_lm_head.weight.data[j, :]
                    i += 1
                j += 1
            model.set_output_embeddings(new_lm_head)

    # Update base model and current model config
    model.config.vocab_size = new_num_tokens
    model.vocab_size = new_num_tokens

    # Tie weights
    # NOTE: only has effect when model.config.tie_word_embeddings and
    # model.config.tie_encoder_decoder is true
    model.tie_weights()

    # Save new model
    model.save_pretrained(model_name)
    logging.info(model_name + " - num_parameters : " + str(model.num_parameters()))
    logging.info(model_name + " - num_tokens : " + str(len(vocab)))

    # Save vocab
    fw = open(os.path.join(model_name, 'vocab.txt'), 'w')
    for token in vocab:
        fw.write(token+'\n')
    fw.close()

    # Save tokenizer config
    fw = open(os.path.join(model_name, 'tokenizer_config.json'), 'w')
    json.dump({"do_lower_case": False, "model_max_length": 512}, fw)
    fw.close()

    return new_embeddings


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description="reducing transformers size")
    parser.add_argument("--source_model",
                        type=str,
                        required=True,
                        default='',
                        help="The multilingual transformer to start from")
    parser.add_argument("--vocab_file",
                        type=str,
                        required=True,
                        default='vocab_5langs.txt',
                        help="The intended vocabulary file path")
    parser.add_argument("--output_model",
                        type=str,
                        required=True,
                        default='output_model',
                        help="The name of the final reduced model")
    parser.add_argument("--convert_to_tf",
                        type=str2bool,
                        required=False,
                        default=False,
                        help="Whether to generate a tenserflow version or not")

    args = parser.parse_args()

    # Load original tokenizer, model and vocab
    logging.info('starting from model: ' + args.source_model)
    if "mt5" in args.source_model:
        tokenizer = T5Tokenizer.from_pretrained(args.source_model)
        model = MT5ForConditionalGeneration.from_pretrained(args.source_model)
        model_type = "mt5"
    elif "mbart" in args.source_model:
        tokenizer = MBartTokenizer.from_pretrained(args.source_model)
        model = MBartForConditionalGeneration.from_pretrained(args.source_model)
        model_type = "mbart"
    else:
        logging.info("model type not supported...")
        exit()
    vocab = tokenizer.get_vocab()

    logging.info(args.source_model + " - num_parameters : " + str(model.num_parameters()))
    logging.info(args.source_model + " - num_tokens : " + str(len(vocab)))

    # Load new vocab
    new_vocab = open(args.vocab_file).read().splitlines()

    # TODO retrain tokenizer from corpus...
    # ...

    # Rebuild pytorch model
    new_embs = select_embeddings(model, vocab, new_vocab, model_type, args.output_model)

    # convert to tensorflow
    if (args.convert_to_tf):
        if os.path.isfile(f"{args.output_model}/tf_model.h5"):
            logging.info(f"{args.output_model}/tf_model.h5 already exists")
        else:
            tf_model = TFAutoModel.from_pretrained(args.output_model, from_pt=True)
            tf_model.save_pretrained(args.output_model)


if __name__ == "__main__":
    main()
