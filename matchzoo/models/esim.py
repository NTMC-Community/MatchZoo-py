"""An implementation of ESIM Model."""
import typing

import torch
import torch.nn as nn
from torch.nn import functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.modules import RNNDropout
from matchzoo.modules import BidirectionalAttention
from matchzoo.modules import StackedBRNN


class ESIM(BaseModel):
    """
    ESIM Model.

    Examples:
        >>> model = ESIM()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=False
        )
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='dropout', value=0.2,
                         desc="Dropout rate."))
        params.add(Param(name='hidden_size', value=200,
                         desc="Hidden size."))
        params.add(Param(name='lstm_layer', value=1,
                         desc="Number of LSTM layers"))
        params.add(Param(name='drop_lstm', value=False,
                         desc="Whether dropout LSTM."))
        params.add(Param(name='concat_lstm', value=True,
                         desc="Whether concat intermediate outputs."))
        params.add(Param(name='rnn_type', value='lstm',
                         desc="Choose rnn type, lstm or gru."))
        return params

    def build(self):
        """Instantiating layers."""
        rnn_mapping = {'lstm': nn.LSTM, 'gru': nn.GRU}
        self.embedding = self._make_default_embedding_layer()
        self.rnn_dropout = RNNDropout(p=self._params['dropout'])
        lstm_size = self._params['hidden_size']
        if self._params['concat_lstm']:
            lstm_size /= self._params['lstm_layer']
        self.input_encoding = StackedBRNN(
            self._params['embedding_output_dim'],
            int(lstm_size / 2),
            self._params['lstm_layer'],
            dropout_rate=self._params['dropout'],
            dropout_output=self._params['drop_lstm'],
            rnn_type=rnn_mapping[self._params['rnn_type'].lower()],
            concat_layers=self._params['concat_lstm'])
        self.attention = BidirectionalAttention()
        self.projection = nn.Sequential(
            nn.Linear(
                4 * self._params['hidden_size'],
                self._params['hidden_size']),
            nn.ReLU())
        self.composition = StackedBRNN(
            self._params['hidden_size'],
            int(lstm_size / 2),
            self._params['lstm_layer'],
            dropout_rate=self._params['dropout'],
            dropout_output=self._params['drop_lstm'],
            rnn_type=rnn_mapping[self._params['rnn_type'].lower()],
            concat_layers=self._params['concat_lstm'])
        self.classification = nn.Sequential(
            nn.Dropout(
                p=self._params['dropout']),
            nn.Linear(
                4 * self._params['hidden_size'],
                self._params['hidden_size']),
            nn.Tanh(),
            nn.Dropout(
                p=self._params['dropout']))
        self.out = self._make_output_layer(self._params['hidden_size'])

    def forward(self, inputs):
        """Forward."""
        # Scalar dimensions referenced here:
        # B = batch size (number of sequences)
        # D = embedding size
        # L = `input_left` sequence length
        # R = `input_right` sequence length
        # H = hidden size

        # [B, L], [B, R]
        query, doc = inputs['text_left'].long(), inputs['text_right'].long()

        # [B, L]
        # [B, R]
        query_mask = (query == self._params['mask_value'])
        doc_mask = (doc == self._params['mask_value'])

        # [B, L, D]
        # [B, R, D]
        query = self.embedding(query)
        doc = self.embedding(doc)

        # [B, L, D]
        # [B, R, D]
        query = self.rnn_dropout(query)
        doc = self.rnn_dropout(doc)

        # [B, L, H]
        # [B, R, H]
        query = self.input_encoding(query, query_mask)
        doc = self.input_encoding(doc, doc_mask)

        # [B, L, H], [B, L, H]
        attended_query, attended_doc = self.attention(
            query, query_mask, doc, doc_mask)

        # [B, L, 4 * H]
        # [B, L, 4 * H]
        enhanced_query = torch.cat([query,
                                    attended_query,
                                    query - attended_query,
                                    query * attended_query],
                                   dim=-1)
        enhanced_doc = torch.cat([doc,
                                  attended_doc,
                                  doc - attended_doc,
                                  doc * attended_doc],
                                 dim=-1)
        # [B, L, H]
        # [B, L, H]
        projected_query = self.projection(enhanced_query)
        projected_doc = self.projection(enhanced_doc)

        # [B, L, H]
        # [B, L, H]
        query = self.composition(projected_query, query_mask)
        doc = self.composition(projected_doc, doc_mask)

        # [B, L]
        # [B, R]
        reverse_query_mask = 1. - query_mask.float()
        reverse_doc_mask = 1. - doc_mask.float()

        # [B, H]
        # [B, H]
        query_avg = torch.sum(query * reverse_query_mask.unsqueeze(2), dim=1)\
            / (torch.sum(reverse_query_mask, dim=1, keepdim=True) + 1e-8)
        doc_avg = torch.sum(doc * reverse_doc_mask.unsqueeze(2), dim=1)\
            / (torch.sum(reverse_doc_mask, dim=1, keepdim=True) + 1e-8)

        # [B, L, H]
        # [B, L, H]
        query = query.masked_fill(query_mask.unsqueeze(2), -1e7)
        doc = doc.masked_fill(doc_mask.unsqueeze(2), -1e7)

        # [B, H]
        # [B, H]
        query_max, _ = query.max(dim=1)
        doc_max, _ = doc.max(dim=1)

        # [B, 4 * H]
        v = torch.cat([query_avg, query_max, doc_avg, doc_max], dim=-1)

        # [B, H]
        hidden = self.classification(v)

        # [B, num_classes]
        out = self.out(hidden)

        return out
