import os
from models.modeling_nezha import (
                            NeZhaConfig, NeZhaForSequenceClassification, NeZhaForTokenClassification,
                            NeZhaBiLSTMForTokenClassification, NeZhaForDocumentClassification,
                            NeZhaForDocumentTagClassification, NeZhaForTagClassification
                        )


pretrained_local_mapping = {
    "facebook/mbart-large-cc25": "",
    "google/mt5-base": "/mnt/dl/public/pretrained_models/mt5-base",
    "google/mt5-large": "/mnt/dl/public/pretrained_models/mt5-large"
}


def get_tokenizer(args):
    if args.model_name == "nezha":
        if args.bert_model:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.bert_model, 'vocab.txt'), do_lower_case=True)
        elif args.trained_model_dir:
            tokenizer = tokenization.BertTokenizer(vocab_file=os.path.join(args.trained_model_dir, 'vocab.txt'), do_lower_case=True)
        else:
            logger.error("BERT vocab file not set, please check your ber_model_dir or trained_model_dir")
        logger.info('vocab size is %d' % (len(tokenizer.vocab)))
        return tokenizer
    elif args.model_name == "bart" or args.model_name == "mbart":
        from models.tokenization_mbart import MBartTokenizer
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
        return tokenizer
    elif args.model_name == "t5" or args.model_name == "mt5":
        from models.tokenization_t5 import T5Tokenizer
        pretrained_tag = "/mnt/dl/public/pretrained_models/mt5-large"
        tokenizer = T5Tokenizer.from_pretrained(pretrained_tag)
        return tokenizer
    else:
        logger.error("can not find the proper tokenizer type...")
        return None


def get_tokenizer_and_model(args, bert_config, label_map, num_labels):
    if args.task_name == "ner":
        if args.ner_addBilstm:
            logger.info('Use BiLSTM in NER Model.')
            return NeZhaBiLSTMForTokenClassification(bert_config, label_map, num_labels=num_labels)
        else:
            return NeZhaForTokenClassification(bert_config, num_labels=num_labels)
    elif args.task_name == "textclf":
        if args.model_name == "nezha":
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
        if args.model_name == "bart" or args.model_name == "mbart":
            from models.modeling_mbart import MBartForConditionalGeneration
            gradient_checkpointing_flag = True if args.gradient_checkpointing else False
            if gradient_checkpointing_flag:
                logger.info("gradient checkpointing enabled")
            model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25", gradient_checkpointing=gradient_checkpointing_flag)
            if args.freeze_encoder:
                model.freeze_encoder()
                model.unfreeze_encoder_last_layers()
            return model
        elif args.model_name == "bart" or args.model_name == "mbart":
            from models.tokenization_t5 import T5Tokenizer
            from models.modeling_mt5 import MT5ForConditionalGeneration
            pretrained_tag = "/mnt/dl/public/pretrained_models/mt5-base"
            model = MT5ForConditionalGeneration.from_pretrained(pretrained_tag)
            if args.freeze_encoder:
                model.freeze_encoder()
                model.unfreeze_encoder_last_layers()
            return model
    elif args.task_name == "translation":
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


def save_model(args, tokenizer, model, epoch=None, step=None):
    if epoch is not None:
        assert isinstance(epoch, int) and isinstance(step, int), "type error"
        sub_dir = "models_ep_" + str(epoch) + "_step_" + str(step)
    else:
        sub_dir = "models"
    if args.model_name == "nezha":
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, sub_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, sub_dir, "bert_config.json")
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
    else:
        saved_path = os.path.join(args.output_dir, sub_dir)
        tokenizer.save_pretrained(saved_path)
        model.save_pretrained(saved_path)
