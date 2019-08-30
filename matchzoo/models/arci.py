"""An implementation of ArcI Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class ArcI(BaseModel):
    """
    ArcI Model.

    Examples:
        >>> model = ArcI()
        >>> model.params['left_filters'] = [32]
        >>> model.params['right_filters'] = [32]
        >>> model.params['left_kernel_sizes'] = [3]
        >>> model.params['right_kernel_sizes'] = [3]
        >>> model.params['left_pool_sizes'] = [2]
        >>> model.params['right_pool_sizes'] = [4]
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 64
        >>> model.params['mlp_num_fan_out'] = 32
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.params['dropout_rate'] = 0.5
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
        params.add(Param(name='left_length', value=10,
                         desc='Length of left input.'))
        params.add(Param(name='right_length', value=100,
                         desc='Length of right input.'))
        params.add(Param(name='conv_activation_func', value='relu',
                         desc="The activation function in the "
                         "convolution layer."))
        params.add(Param(name='left_filters', value=[32],
                         desc="The filter size of each convolution "
                         "blocks for the left input."))
        params.add(Param(name='left_kernel_sizes', value=[3],
                         desc="The kernel size of each convolution "
                         "blocks for the left input."))
        params.add(Param(name='left_pool_sizes', value=[2],
                         desc="The pooling size of each convolution "
                         "blocks for the left input."))
        params.add(Param(name='right_filters', value=[32],
                         desc="The filter size of each convolution "
                         "blocks for the right input."))
        params.add(Param(name='right_kernel_sizes', value=[3],
                         desc="The kernel size of each convolution "
                         "blocks for the right input."))
        params.add(Param(name='right_pool_sizes', value=[2],
                         desc="The pooling size of each convolution "
                         "blocks for the right input."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = 10,
        fixed_length_right: int = 100,
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
        """
        Build model structure.

        ArcI use Siamese arthitecture.
        """
        self.embedding = self._make_default_embedding_layer()

        # Build conv
        activation = parse_activation(self._params['conv_activation_func'])
        left_in_channels = [
            self._params['embedding_output_dim'],
            *self._params['left_filters'][:-1]
        ]
        right_in_channels = [
            self._params['embedding_output_dim'],
            *self._params['right_filters'][:-1]
        ]
        conv_left = [
            self._make_conv_pool_block(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(left_in_channels,
                                      self._params['left_filters'],
                                      self._params['left_kernel_sizes'],
                                      self._params['left_pool_sizes'])
        ]
        conv_right = [
            self._make_conv_pool_block(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(right_in_channels,
                                      self._params['right_filters'],
                                      self._params['right_kernel_sizes'],
                                      self._params['right_pool_sizes'])
        ]
        self.conv_left = nn.Sequential(*conv_left)
        self.conv_right = nn.Sequential(*conv_right)

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

        left_length = self._params['left_length']
        right_length = self._params['right_length']
        for ps in self._params['left_pool_sizes']:
            left_length = left_length // ps
        for ps in self._params['right_pool_sizes']:
            right_length = right_length // ps
        self.mlp = self._make_multi_layer_perceptron_layer(
            left_length * self._params['left_filters'][-1] + (
                right_length * self._params['right_filters'][-1])
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
        #   F = number of filters
        #   P = pool size

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # Process left and right input.
        # shape = [B, D, L]
        # shape = [B, D, R]
        embed_left = self.embedding(input_left.long()).transpose(1, 2)
        embed_right = self.embedding(input_right.long()).transpose(1, 2)

        # Convolution
        # shape = [B, F, L // P]
        # shape = [B, F, R // P]
        conv_left = self.conv_left(embed_left)
        conv_right = self.conv_right(embed_right)

        # shape = [B, F * (L // P)]
        # shape = [B, F * (R // P)]
        rep_left = torch.flatten(conv_left, start_dim=1)
        rep_right = torch.flatten(conv_right, start_dim=1)

        # shape = [B, F * (L // P) + F * (R // P)]
        concat = self.dropout(torch.cat((rep_left, rep_right), dim=1))

        # shape = [B, *]
        dense_output = self.mlp(concat)

        out = self.out(dense_output)
        return out

    @classmethod
    def _make_conv_pool_block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: nn.Module,
        pool_size: int,
    ) -> nn.Module:
        """Make conv pool block."""
        return nn.Sequential(
            nn.ConstantPad1d((0, kernel_size - 1), 0),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool1d(kernel_size=pool_size)
        )
