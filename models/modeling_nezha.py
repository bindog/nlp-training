from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from .configuration_bert import BertConfig
from .modeling_bert import (BertIntermediate, BertOutput, BertPooler,
                            BertPreTrainedModel, BertSelfOutput, BertSelfAttention,
                            BertOnlyMLMHead, BertPreTrainingHeads)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(filename)s %(lineno)d] %(message)s")
logger = logging.getLogger(__name__)


NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Noahs-Ark/nezha-base-en",
    "Noahs-Ark/nezha-base-cn",
    "Noahs-Ark/nezha-large-cn",
]


class NeZhaConfig(BertConfig):
    """Configuration class to store the configuration of a `NeZhaModel`.
    """
    def __init__(self, use_relative_position=True, max_relative_position=64, **kwargs):
        super().__init__(**kwargs)
        self.use_relative_position = use_relative_position
        self.max_relative_position = max_relative_position


class NeZhaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(NeZhaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        try:
            self.use_relative_position = config.use_relative_position
        except:
            self.use_relative_position = False
        if not self.use_relative_position:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if not self.use_relative_position:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def _generate_relative_positions_matrix(length, max_relative_position,
                                        cache=False):
    """Generates matrix of relative positions between inputs."""
    if not cache:
        range_vec = torch.arange(length)
        range_mat = range_vec.repeat(length).view(length, length)
        distance_mat = range_mat - torch.t(range_mat)
    else:
        distance_mat = torch.arange(-length + 1, 1, 1).unsqueeze(0)

    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length).view(length, length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = np.zeros([vocab_size, depth])
    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = torch.tensor(embeddings_table).float()
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    embeddings = embeddings.view(my_shape)
    return embeddings


class NeZhaSelfAttention(nn.Module):
    def __init__(self, config):
        super(NeZhaSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.relative_positions_embeddings = _generate_relative_positions_embeddings(
            length=512, depth=self.attention_head_size, max_relative_position=config.max_relative_position).cuda()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        device = 'cpu'
        if hidden_states.is_cuda:
            device = hidden_states.get_device()
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        batch_size, num_attention_heads, from_seq_length, to_seq_length = attention_scores.size()

        relations_keys = self.relative_positions_embeddings.detach().clone()[:to_seq_length, :to_seq_length, :].to(
            device)
        # relations_keys = embeddings.clone().detach().to(device)
        query_layer_t = query_layer.permute(2, 0, 1, 3)
        query_layer_r = query_layer_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                        self.attention_head_size)
        key_position_scores = torch.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
        key_position_scores_r = key_position_scores.view(from_seq_length, batch_size,
                                                         num_attention_heads, from_seq_length)
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        relations_values = self.relative_positions_embeddings.clone()[:to_seq_length, :to_seq_length, :].to(
            device)
        attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        attentions_probs_r = attention_probs_t.contiguous().view(from_seq_length, batch_size * num_attention_heads,
                                                                 to_seq_length)
        value_position_scores = torch.matmul(attentions_probs_r, relations_values)
        value_position_scores_r = value_position_scores.view(from_seq_length, batch_size,
                                                             num_attention_heads, self.attention_head_size)
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        context_layer = context_layer + value_position_scores_r_t

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype


class NeZhaAttention(nn.Module):
    def __init__(self, config):
        super(NeZhaAttention, self).__init__()
        try:
            self.use_relative_position = config.use_relative_position
        except:
            self.use_relative_position = False
        if self.use_relative_position:
            self.self = NeZhaSelfAttention(config)
        else:
            self.self = BertSelfAttention(config)

        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        self_output, layer_att = self_output
        attention_output = self.output(self_output, input_tensor)
        return attention_output, layer_att


class NeZhaLayer(nn.Module):
    def __init__(self, config):
        super(NeZhaLayer, self).__init__()
        self.attention = NeZhaAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output, layer_att = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, layer_att


class NeZhaEncoder(nn.Module):
    def __init__(self, config):
        super(NeZhaEncoder, self).__init__()
        layer = NeZhaLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        all_encoder_att = []
        for i, layer_module in enumerate(self.layer):
            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(all_encoder_layers[i], attention_mask)
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)
        all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_att


class NeZhaModel(BertPreTrainedModel):
    def __init__(self, config):
        super(NeZhaModel, self).__init__(config)
        self.embeddings = NeZhaEmbeddings(config)
        self.encoder = NeZhaEncoder(config)
        self.pooler = BertPooler(config)

        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_attention_mask=False,
                model_distillation=False, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask)
        encoded_layers, attention_layers = encoded_layers
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if output_attention_mask:
            return encoded_layers, attention_layers, pooled_output, extended_attention_mask
        if model_distillation:
            return encoded_layers, attention_layers
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class NeZhaForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(NeZhaForPreTraining, self).__init__(config)
        self.bert = NeZhaModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        elif masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            total_loss = masked_lm_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class NeZhaForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(NeZhaForMaskedLM, self).__init__(config)
        self.bert = NeZhaModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                output_att=False, infer=False):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=True, output_att=output_att)

        if output_att:
            sequence_output, att_output = sequence_output
        prediction_scores = self.cls(sequence_output[-1])

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            if not output_att:
                return masked_lm_loss
            else:
                return masked_lm_loss, att_output
        else:
            if not output_att:
                return prediction_scores
            else:
                return prediction_scores, att_output


class NeZhaForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(NeZhaForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        task_output = self.dropout(pooled_output)
        logits = self.classifier(task_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class NeZhaForDocumentClassification(BertPreTrainedModel):
    def __init__(self, config, doc_inner_batch_size, num_labels, bidirectional=True):
        super(NeZhaForDocumentClassification, self).__init__(config)
        self.bert = NeZhaModel(config)
        self.doc_inner_batch_size = doc_inner_batch_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, bidirectional=bidirectional)
        hidden_dimension = config.hidden_size * 2 if bidirectional else config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(hidden_dimension, num_labels),
        )

    def forward(self, document_compose, labels=None):
        '''
        Args:
            document_compose: the output of the encode document function,
                            shape of document_compose is [num_docs, max_sequences_per_document, 3, max_seq_length=128],
                            dimension 3 means input_ids, token_type_ids, attention_masks
        '''
        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(
                                size=(
                                    document_compose.shape[0],
                                    self.doc_inner_batch_size,
                                    self.bert.config.hidden_size
                                ),
                                dtype=torch.float).cuda()

        # only pass through doc_inner_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_compose.shape[0]):
            _, pooled_output = self.bert(
                                    input_ids=document_compose[doc_id][:, 0],
                                    token_type_ids=document_compose[doc_id][:, 1],
                                    attention_mask=document_compose[doc_id][:, 2]
                                )
            bert_output[doc_id] = self.dropout(pooled_output)

        output, (_, _) = self.gru(bert_output.permute(1,0,2))
        last_layer = output[-1]
        logits = self.classifier(last_layer)
        assert logits.shape[0] == document_compose.shape[0]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def freeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class NeZhaForTagClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NeZhaForTagClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            # TODO validation this effect
            # from utils.loss import multilabel_categorical_crossentropy as loss_fct
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class NeZhaForDocumentTagClassification(BertPreTrainedModel):
    def __init__(self, config, doc_inner_batch_size, num_labels):
        super(NeZhaForDocumentTagClassification, self).__init__(config)
        self.bert = NeZhaModel(config)
        self.doc_inner_batch_size = doc_inner_batch_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels),
            nn.Tanh()
        )

    def forward(self, document_compose, labels=None):
        '''
        Args:
            document_compose: the output of the encode document function,
                            shape of document_compose is [num_docs, max_sequences_per_document, 3, max_seq_length=128],
                            dimension 3 means input_ids, token_type_ids, attention_masks
        '''
        # contains all BERT sequences
        # bert should output a (batch_size, num_sequences, bert_hidden_size)
        bert_output = torch.zeros(
                                size=(
                                    document_compose.shape[0],
                                    self.doc_inner_batch_size,
                                    self.bert.config.hidden_size
                                ),
                                dtype=torch.float).cuda()

        # only pass through doc_inner_batch_size numbers of inputs into bert.
        # this means that we are possibly cutting off the last part of documents.
        for doc_id in range(document_compose.shape[0]):
            _, pooled_output = self.bert(
                                    input_ids=document_compose[doc_id][:, 0],
                                    token_type_ids=document_compose[doc_id][:, 1],
                                    attention_mask=document_compose[doc_id][:, 2]
                                )
            bert_output[doc_id] = self.dropout(pooled_output)

        output, (_, _) = self.lstm(bert_output.permute(1,0,2))
        last_layer = output[-1]
        logits = self.classifier(last_layer)
        assert logits.shape[0] == document_compose.shape[0]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class NeZhaForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(NeZhaForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class NeZhaBiLSTMForTokenClassification(BertPreTrainedModel):

    """BERT-BiLSTM-CRF model for NER task.
    This module is composed of the NeZhaForTokenClassification model with a BiLSTM layer and a CRF layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `label_map_reverse`: a map of label_name to label_id.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    label_ids = torch.LongTensor([[16, 1, 0], [16, 3, 0]])

    label_map_reverse = {"I-COM": "15", "[CLS]": "16",}
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2
    model = BertForSequenceClassification(config, label_map_reverse, num_labels)

    # In train step, we use neg_log_likelihood to get loss.
    loss = model.neg_log_likelihood(input_ids, token_type_ids, input_mask, label_ids)

    # In eval step, we use model to get result.
    logits = model(input_ids, segment_ids, input_mask, None)
    ```
    """
    def __init__(self, config, label_map_reverse, num_labels):
        super(NeZhaBiLSTMForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.label_map_reverse = label_map_reverse
        self.hidden_size = config.hidden_size
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bilstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size, hidden_size=config.hidden_size//2, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.start_label_id = self.label_map_reverse['[CLS]']
        self.transitions = nn.Parameter(torch.randn(
            self.num_labels, self.num_labels
        ))
        self.apply(self._init_weights)

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).cuda()
        log_delta[:, 0, self.start_label_id] = 0.

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def log_sum_exp_batch(self, log_Tensor, axis=-1): # shape (batch_size,n,m)
        return torch.max(log_Tensor, axis)[0] + \
            torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).cuda()  #[batch_size, 1, 16]
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (self.log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = self.log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).cuda()
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _get_lstm_features(self, input_ids, token_type_ids, attention_mask):
        """sentence is the ids"""
        # self.hidden = self.init_hidden()
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        rnn_out, _ = self.bilstm(sequence_output)
        sequence_output = self.dropout(rnn_out)
        logits = self.classifier(sequence_output)
        return logits  # [8, 75, 16]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        logits = self._get_lstm_features(input_ids, token_type_ids, attention_mask)  # [8, 180,768]
        loss = None
        if labels is not None:
            forward_score = self._forward_alg(logits)
            gold_score = self._score_sentence(logits, labels)
            loss = torch.mean(forward_score - gold_score)
            return loss
        else:
            score, logits = self._viterbi_decode(logits)
            return logits

class NeZhaForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config, num_choices=2):
        super(NeZhaForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, return_logits=False):
        # input_ids: [bs,num_choice,seq_l]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))  # flat_input_ids: [bs*num_choice,seq_l]
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
        reshaped_logits = logits.view(-1, self.num_choices)  # logits: (bs, num_choice)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            if return_logits:
                return loss, reshaped_logits
            else:
                return loss
        else:
            return reshaped_logits


class NeZhaForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(NeZhaForQuestionAnswering, self).__init__(config)
        self.bert = NeZhaModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class NeZhaForJointLSTM(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels):
        super(NeZhaForJointLSTM, self).__init__(config)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.bert = NeZhaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=300,
            batch_first=True,
            bidirectional=True
        )
        self.slot_classifier = nn.Linear(300 * 2, num_slot_labels)
        self.apply(self._init_weights)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, intent_labels=None, slot_labels=None):
        encoded_layers, attention_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        intent_logits = self.intent_classifier(self.dropout(pooled_output))

        last_encoded_layer = encoded_layers[-1]
        slot_logits, _ = self.lstm(last_encoded_layer)
        slot_logits = self.slot_classifier(slot_logits)
        tmp = []
        if intent_labels is not None and slot_labels is not None:
            loss_fct = CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            if attention_mask is not None:
                active_slot_loss = attention_mask.view(-1) == 1
                active_slot_logits = slot_logits.view(-1, self.num_slot_labels)[active_slot_loss]
                active_slot_labels = slot_labels.view(-1)[active_slot_loss]
                slot_loss = loss_fct(active_slot_logits, active_slot_labels)
            else:
                slot_loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))

            return intent_loss, slot_loss
        else:
            return intent_logits, slot_logits
