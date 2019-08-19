"""An implementation of Match LSTM Model."""
import typing

import torch
import torch.nn as nn
from torch.nn import functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.modules import MatchModule
from matchzoo.modules import StackedBRNN


class MatchLSTM(BaseModel):
    """
    MatchLSTM Model.

    https://github.com/shuohangwang/mprc/blob/master/qa/rankerReader.lua.

    Examples:
        >>> model = MatchLSTM()
        >>> model.params['dropout'] = 0.2
        >>> model.params['hidden_size'] = 200
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
        self.dropout = nn.Dropout(p=self._params['dropout'])
        if self._params['concat_lstm']:
            lstm_layer = self._params['lstm_layer']
            lstm_size = self._params['hidden_size'] / lstm_layer
        self.input_proj = StackedBRNN(
            self._params['embedding_output_dim'],
            int(lstm_size / 2),
            self._params['lstm_layer'],
            dropout_rate=self._params['dropout'],
            dropout_output=self._params['drop_lstm'],
            rnn_type=rnn_mapping[self._params['rnn_type'].lower()],
            concat_layers=self._params['concat_lstm'])
        self.match_module = MatchModule(
            self._params['hidden_size'], dropout_rate=self._params['dropout'])
        self.mlstm_module = StackedBRNN(
            2 * self._params['hidden_size'],
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
                self._params['hidden_size'],
                self._params['hidden_size']),
            nn.Tanh())
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
        query = self.dropout(query)
        doc = self.dropout(doc)

        # [B, L, H]
        # [B, R, H]
        query = self.input_proj(query, query_mask)
        doc = self.input_proj(doc, doc_mask)

        # [B, L, H]
        match_out = self.match_module(
            query, doc, doc_mask)

        # [B, L, H]
        mlstm_out = self.mlstm_module(match_out, query_mask)

        # [B, H]
        max_pool_rep, _ = mlstm_out.max(dim=1)

        # [B, H]
        hidden = self.classification(max_pool_rep)

        # [B, num_classes]
        out = self.out(hidden)

        return out
