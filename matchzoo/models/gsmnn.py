"""An implementation of GSMNN Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class GSMNN(BaseModel):
    """
    GSMNN Model.

    Examples:
        >>> model = GSMNN()
        >>> model.params['embedding_input_dim'] = 200
        >>> model.params['embedding_output_dim'] = 50
        >>> model.params['left_length'] = 30
        >>> model.params['right_length'] = 30
        >>> model.params['filters'] = 32
        >>> model.params['conv_activation_func'] = 'tanh'
        >>> model.params['max_ngram'] = 3
        >>> model.params['bidirection'] = False
        >>> model.params['row_hidden_size'] = 100
        >>> model.params['column_hidden_size'] = 100
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(
            name='left_length',
            value=30,
            desc="Length of left input."
        ))
        params.add(Param(
            name='right_length',
            value=30,
            desc="Length of right input."
        ))
        params.add(Param(
            name='filters',
            value=32,
            desc="The filter size in the convolution layer."
        ))
        params.add(Param(
            name='conv_activation_func',
            value='tanh',
            desc="The activation function in the convolution layer."
        ))
        params.add(Param(
            name='max_ngram',
            value=3,
            desc="The maximum length of n-grams for the convolution layer."
        ))
        params.add(Param(
            name='bidirection',
            value=False,
            desc="Whether use bidirection in row LSTM and column LSTM."
        ))
        params.add(Param(
            name='row_hidden_size',
            value=100,
            desc="The hidden size of row LSTM layer."
        ))
        params.add(Param(
            name='column_hidden_size',
            value=100,
            desc="The hidden size of column LSTM layer."
        ))
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = 30,
        fixed_length_right: int = 30,
        pad_word_value: typing.Union[int, str] = 0,
        pad_word_mode: str = 'pre',
        with_ngram: bool = False,
        fixed_ngram_length: int = None,
        pad_ngram_value: typing.Union[int, str] = 0,
        pad_ngram_mode: str = 'pre'
    ) -> BaseCallback:
        """
        Model default padding callback.

        The padding callback's on_batch_unpacked would pad a batch of data to
        a fixed length.

        :return: Default padding callback.
        """
        return callbacks.BasicPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_word_value=pad_word_value,
            pad_word_mode=pad_word_mode,
            with_ngram=with_ngram,
            fixed_ngram_length=fixed_ngram_length,
            pad_ngram_value=pad_ngram_value,
            pad_ngram_mode=pad_ngram_mode
        )

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()

        self.convs = nn.ModuleList()
        q_conv_out_size = 0
        d_conv_out_size = 0
        for i in range(self._params['max_ngram']):
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=self._params['embedding_output_dim'],
                    out_channels=self._params['filters'],
                    kernel_size=i + 1
                ),
                parse_activation(self._params['conv_activation_func'])
            )
            self.convs.append(conv)
            q_conv_out_size += (self._params['left_length'] - i)
            d_conv_out_size += (self._params['right_length'] - i)

        self.row_lstm = nn.LSTM(
            input_size=d_conv_out_size,
            hidden_size=self._params['row_hidden_size'],
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=self._params['bidirection']
        )
        self.col_lstm = nn.LSTM(
            input_size=q_conv_out_size,
            hidden_size=self._params['column_hidden_size'],
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=self._params['bidirection']
        )
        row_in_feature = (
            self._params['row_hidden_size']
            if not self._params['bidirection']
            else self._params['row_hidden_size'] * 2
        )
        col_in_feature = (
            self._params['column_hidden_size']
            if not self._params['bidirection']
            else self._params['column_hidden_size'] * 2
        )
        self.row_attn_layer = nn.Linear(row_in_feature, 1, bias=True)
        self.col_attn_layer = nn.Linear(col_in_feature, 1, bias=True)

        in_feature = 2 * self._params['max_ngram'] * self._params['filters'] + \
            q_conv_out_size * d_conv_out_size
        self.out = self._make_output_layer(in_feature)

    def forward(self, inputs):
        """Forward."""
        query, doc = inputs['text_left'], inputs['text_right']

        # Word embedding layer.
        q_embed = self.embedding(query.long()).transpose(1, 2)
        d_embed = self.embedding(doc.long()).transpose(1, 2)

        # Sentence representation module.
        q_convs = []
        d_convs = []
        for conv in self.convs:
            q_convs.append(conv(q_embed))
            d_convs.append(conv(d_embed))

        q_sent_repre = torch.cat(
            [torch.max(q_conv, dim=-1)[0] for q_conv in q_convs], dim=-1)
        d_sent_repre = torch.cat(
            [torch.max(d_conv, dim=-1)[0] for d_conv in d_convs], dim=-1)

        q_gran_repre = torch.cat(
            [torch.max(q_conv, dim=1)[0] for q_conv in q_convs], dim=-1)
        d_gran_repre = torch.cat(
            [torch.max(d_conv, dim=1)[0] for d_conv in d_convs], dim=-1)

        # Sentence-level matching layer.
        q_d_minus = q_sent_repre - d_sent_repre
        q_d_mul = q_sent_repre * d_sent_repre
        sent_matching = torch.cat([q_d_minus, q_d_mul], dim=-1)

        # Multi-granular matching layer.
        gran_matching = torch.einsum('bl,br->blr', q_gran_repre, d_gran_repre)

        row_encoding, _ = self.row_lstm(gran_matching)
        col_encoding, _ = self.col_lstm(gran_matching.transpose(1, 2))
        row_attn = F.softmax(
            self.row_attn_layer(row_encoding).squeeze(dim=-1), dim=-1)
        col_attn = F.softmax(
            self.col_attn_layer(col_encoding).squeeze(dim=-1), dim=-1)
        attn_matrix = torch.einsum('bl,br->blr', row_attn, col_attn)

        attn_gran_matching = gran_matching * attn_matrix
        attn_gran_matching = torch.flatten(attn_gran_matching, start_dim=1)

        # Global matching layer.
        global_matching = torch.cat([sent_matching, attn_gran_matching], dim=-1)
        output = self.out(global_matching)

        return output
