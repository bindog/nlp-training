import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class SimpleFC(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super(SimpleFC, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_labels)
        self.num_labels = num_labels

    def forward(self, input_embeddings, labels=None):
        logits = self.fc(input_embeddings)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return loss
        else:
            return logits
