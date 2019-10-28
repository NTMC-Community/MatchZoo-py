"""An implementation of MVLSTM Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks


class MVLSTM(BaseModel):
    """
    MVLSTM Model.

    Examples:
        >>> model = MVLSTM()
        >>> model.params['hidden_size'] = 32
        >>> model.params['top_k'] = 50
        >>> model.params['mlp_num_layers'] = 2
        >>> model.params['mlp_num_units'] = 20
        >>> model.params['mlp_num_fan_out'] = 10
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['dropout_rate'] = 0.0
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=True,
            with_multi_layer_perceptron=True
        )
        params.add(Param(name='hidden_size', value=32,
                         desc="Integer, the hidden size in the "
                              "bi-directional LSTM layer."))
        params.add(Param(name='num_layers', value=1,
                         desc="Integer, number of recurrent layers."))
        params.add(Param(
            'top_k', value=10,
            hyper_space=hyper_spaces.quniform(low=2, high=100),
            desc="Size of top-k pooling layer."
        ))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="Float, the dropout rate."
        ))
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = 10,
        fixed_length_right: int = 40,
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

        self.left_bilstm = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['hidden_size'],
            num_layers=self._params['num_layers'],
            batch_first=True,
            dropout=self._params['dropout_rate'],
            bidirectional=True
        )
        self.right_bilstm = nn.LSTM(
            input_size=self._params['embedding_output_dim'],
            hidden_size=self._params['hidden_size'],
            num_layers=self._params['num_layers'],
            batch_first=True,
            dropout=self._params['dropout_rate'],
            bidirectional=True
        )

        self.mlp = self._make_multi_layer_perceptron_layer(
            self._params['top_k']
        )
        self.dropout = nn.Dropout(p=self._params['dropout_rate'])
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
        #   H = LSTM hidden size
        #   K = size of top-k

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        query, doc = inputs['text_left'], inputs['text_right']

        # Process left and right input.
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_query = self.embedding(query.long())
        embed_doc = self.embedding(doc.long())

        # Bi-directional LSTM
        # shape = [B, L, 2 * H]
        # shape = [B, R, 2 * H]
        rep_query, _ = self.left_bilstm(embed_query)
        rep_doc, _ = self.right_bilstm(embed_doc)

        # Top-k matching
        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(rep_query, p=2, dim=-1),
            F.normalize(rep_doc, p=2, dim=-1)
        )
        # shape = [B, L * R]
        matching_signals = torch.flatten(matching_matrix, start_dim=1)
        # shape = [B, K]
        matching_topk = torch.topk(
            matching_signals,
            k=self._params['top_k'],
            dim=-1,
            sorted=True
        )[0]

        # shape = [B, *]
        dense_output = self.mlp(matching_topk)

        # shape = [B, *]
        out = self.out(self.dropout(dense_output))
        return out
