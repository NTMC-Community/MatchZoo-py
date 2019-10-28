"""An implementation of HBMP Model."""
import typing

import copy
import torch
import torch.nn as nn

from matchzoo.engine import hyper_spaces
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel


class HBMP(BaseModel):
    """
    HBMP model.

    Examples:
        >>> model = HBMP()
        >>> model.params['embedding_input_dim'] = 200
        >>> model.params['embedding_output_dim'] = 100
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 10
        >>> model.params['mlp_num_fan_out'] = 10
        >>> model.params['mlp_activation_func'] = nn.LeakyReLU(0.1)
        >>> model.params['lstm_hidden_size'] = 5
        >>> model.params['lstm_num'] = 3
        >>> model.params['num_layers'] = 3
        >>> model.params['dropout_rate'] = 0.1
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True, with_multi_layer_perceptron=True)
        params.add(Param(name='lstm_hidden_size', value=5,
                         desc="Integer, the hidden size of the "
                              "bi-directional LSTM layer."))
        params.add(Param(name='lstm_num', value=3,
                         desc="Integer, number of LSTM units"))
        params.add(Param(name='num_layers', value=1,
                         desc="Integer, number of LSTM layers."))
        params.add(Param(
            name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        HBMP use Siamese arthitecture.
        """
        self.embedding = self._make_default_embedding_layer()

        encoder_layer = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['lstm_hidden_size'],
            num_layers=self._params['num_layers'],
            dropout=self._params['dropout_rate'],
            batch_first=True,
            bidirectional=True)
        self.encoder = nn.ModuleList(
            [copy.deepcopy(encoder_layer)
             for _ in range(self._params['lstm_num'])])
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.mlp = self._make_multi_layer_perceptron_layer(
            self._params['lstm_hidden_size'] * 24
        )

        self.out = self._make_output_layer(
            self._params['mlp_num_fan_out']
        )

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   F = hidden size of LSTM layer

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']
        batch_size = input_left.shape[0]
        state_shape = (
            2 * self._params['num_layers'], batch_size,
            self._params['lstm_hidden_size']
        )

        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        out_left = []
        h_left = c_left = torch.zeros(*state_shape, device=input_left.device)
        for layer in self.encoder:
            out, (h_left, c_left) = layer(embed_left, (h_left, c_left))
            out_left.append(self.max_pool(out.transpose(1, 2)).squeeze(2))

        out_right = []
        h_right = c_right = torch.zeros(*state_shape, device=input_left.device)
        for layer in self.encoder:
            out, (h_right, c_right) = layer(embed_right, (h_right, c_right))
            out_right.append(self.max_pool(out.transpose(1, 2)).squeeze(2))

        # shape = [B, 6 * F]
        encode_left = torch.cat(out_left, 1)
        encode_right = torch.cat(out_right, 1)

        embed_minus = torch.abs(encode_left - encode_right)
        embed_multiply = encode_left * encode_right
        encode_concat = torch.cat(
            [encode_left, encode_right, embed_minus, embed_multiply], 1)

        output = self.mlp(encode_concat)
        output = self.out(output)

        return output
