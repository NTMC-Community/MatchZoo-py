"""An implementation of DUET Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import Attention
from matchzoo.utils import parse_activation
from matchzoo import preprocessors


class DUET(BaseModel):
    """
    Duet Model.

    Examples:
        >>> model = DUET()
        >>> model.params['left_length'] = 10
        >>> model.params['right_length'] = 100
        >>> model.params['lm_filters'] = 300
        >>> model.params['mlp_num_layers'] = 2
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['mlp_num_fan_out'] = 300
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['vocab_size'] = 2000
        >>> model.params['dm_filters'] = 300
        >>> model.params['dm_conv_activation_func'] = 'relu'
        >>> model.params['dm_kernel_size'] = 3
        >>> model.params['dm_right_pool_size'] = 8
        >>> model.params['dropout_rate'] = 0.5
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(
            with_embedding=False,
            with_multi_layer_perceptron=True
        )
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='left_length', value=5,
                         desc='Length of left input.'))
        params.add(Param(name='right_length', value=20,
                         desc='Length of right input.'))
        params.add(Param(name='lm_filters', value=300,
                         desc="Filter size of 1D convolution layer in "
                              "the local model."))
        params.add(Param(name='vocab_size', value=2000,
                         desc="Vocabulary size of the tri-letters used in "
                              "the distributed model."))
        params.add(Param(name='dm_filters', value=300,
                         desc="Filter size of 1D convolution layer in "
                              "the distributed model."))
        params.add(Param(name='dm_kernel_size', value=3,
                         desc="Kernel size of 1D convolution layer in "
                              "the distributed model."))
        params.add(Param(name='dm_conv_activation_func', value='relu',
                         desc="Activation functions of the convolution layer "
                              "in the distributed model."))
        params.add(Param(name='dm_right_pool_size', value=100,
                         desc="Kernel size of 1D convolution layer in "
                              "the distributed model."))
        params.add(Param(
            name='dropout_rate', value=0.5,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8, q=0.02),
            desc="The dropout rate."))
        return params

    @classmethod
    def get_default_preprocessor(
        cls,
        truncated_mode: str = 'pre',
        truncated_length_left: int = 30,
        truncated_length_right: int = 30,
        filter_mode: str = 'df',
        filter_low_freq: float = 1,
        filter_high_freq: float = float('inf'),
        remove_stop_words: bool = False,
        ngram_size: int = 3
    ):
        """:return: Default preprocessor."""
        return preprocessors.BasicPreprocessor(
            truncated_mode=truncated_mode,
            truncated_length_left=truncated_length_left,
            truncated_length_right=truncated_length_right,
            filter_mode=filter_mode,
            filter_low_freq=filter_low_freq,
            filter_high_freq=filter_high_freq,
            remove_stop_words=remove_stop_words,
            ngram_size=ngram_size
        )

    @classmethod
    def _xor_match(cls, x, y):
        """Xor match of two inputs."""
        x_expand = torch.unsqueeze(x, 2).repeat(1, 1, y.shape[1])
        y_expand = torch.unsqueeze(y, 1).repeat(1, x.shape[1], 1)
        out = torch.eq(x_expand, y_expand).float()
        return out

    def build(self):
        """Build model structure."""
        self.lm_conv1d = nn.Conv1d(
            in_channels=self._params['right_length'],
            out_channels=self.params['lm_filters'],
            kernel_size=1,
            stride=1
        )
        lm_mlp_size = self._params['left_length'] * self._params['lm_filters']
        self.lm_mlp = self._make_multi_layer_perceptron_layer(lm_mlp_size)
        self.lm_linear = self._make_perceptron_layer(
            in_features=self._params['mlp_num_fan_out'],
            out_features=1
        )

        self.dm_conv_activation_func = parse_activation(
                self._params['dm_conv_activation_func']
        )
        self.dm_conv_left = nn.Conv1d(self._params['vocab_size'],
                                      self._params['dm_filters'],
                                      self._params['dm_kernel_size']
                                      )
        self.dm_mlp_left = self._make_perceptron_layer(
            in_features=self._params['dm_filters'],
            out_features=self._params['dm_filters']
        )
        self.dm_conv1_right = nn.Conv1d(self._params['vocab_size'],
                                        self._params['dm_filters'],
                                        self._params['dm_kernel_size']
                                        )
        self.dm_conv2_right = nn.Conv1d(self._params['dm_filters'],
                                        self._params['dm_filters'],
                                        1
                                        )
        dm_mp_size = ((self._params['right_length'] -
            self._params['dm_kernel_size'] + 1) //
            self._params['dm_right_pool_size']) * self._params['dm_filters']
        self.dm_mlp = self._make_multi_layer_perceptron_layer(dm_mp_size)
        self.dm_linear = self._make_perceptron_layer(
            in_features=self._params['mlp_num_fan_out'],
            out_features=1
        )

        self.dropout = nn.Dropout(self._params['dropout_rate'])

        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        query_word, doc_word = inputs['text_left'], inputs['text_right']

        # shape = [B, L]
        mask_query = (query_word != self._params['mask_value']).float()
        mask_doc = (doc_word != self._params['mask_value']).float()

        # shape = [B, ngram_size, L]
        # shape = [B, ngram_size, R]
        query_ngram, doc_ngram = inputs['ngram_left'], inputs['ngram_right']

        query_ngram = F.normalize(query_ngram, p=2, dim=2)
        doc_ngram = F.normalize(doc_ngram, p=2, dim=2)

        # shape = [B, R, L]
        matching_xor =  self._xor_match(doc_word, query_word)
        mask_xor = torch.einsum('bi, bj->bij', mask_doc, mask_query)
        xor_res = torch.einsum('bij, bij->bij', matching_xor, mask_xor)

        # Process local model
        lm_res = self.lm_conv1d(xor_res)
        lm_res = lm_res.flatten(start_dim=1, end_dim=-1)
        lm_res = self.lm_mlp(lm_res)
        lm_res = self.dropout(lm_res)
        lm_res = self.lm_linear(lm_res)

        # Process distributed model
        dm_left = self.dm_conv_left(query_ngram.permute(0, 2, 1))
        dm_left = self.dm_conv_activation_func(dm_left)
        dm_left = torch.max(dm_left, dim=-1)[0]
        dm_left = self.dm_mlp_left(dm_left)

        dm_right = self.dm_conv1_right(doc_ngram.permute(0, 2, 1))
        dm_right = F.max_pool2d(self.dm_conv_activation_func(dm_right),
                (1, self._params['dm_right_pool_size']))
        dm_right = self.dm_conv2_right(dm_right)

        dm_res = torch.einsum('bl,blk->blk', dm_left, dm_right)
        dm_res = dm_res.flatten(start_dim=1, end_dim=-1)
        dm_res = self.dm_mlp(dm_res)
        dm_res = self.dropout(dm_res)
        dm_res = self.dm_linear(dm_res)

        x = lm_res + dm_res

        out = self.out(x.unsqueeze(dim=-1))
        return out
