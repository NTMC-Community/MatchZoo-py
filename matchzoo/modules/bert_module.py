"""Bert module."""
import typing

import torch
import torch.nn as nn
from pytorch_transformers import BertModel


class BertModule(nn.Module):
    """
    Bert module.

    BERT (from Google) released with the paper BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding by Jacob Devlin,
    Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

    :param mode: String, supported mode can be referred
        https://huggingface.co/pytorch-transformers/pretrained_models.html.

    """

    def __init__(self, mode: str = 'bert-base-uncased'):
        """:class:`BertModule` constructor."""
        super().__init__()
        self.bert = BertModel.from_pretrained(mode)

    def forward(self, x, y):
        """Forward."""
        inputs = torch.cat((x, y), dim=-1)
        return self.bert(inputs)
