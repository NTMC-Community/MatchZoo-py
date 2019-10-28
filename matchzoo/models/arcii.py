"""An implementation of ArcII Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import Matching
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class ArcII(BaseModel):
    """
    ArcII Model.

    Examples:
        >>> model = ArcII()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['kernel_1d_count'] = 32
        >>> model.params['kernel_1d_size'] = 3
        >>> model.params['kernel_2d_count'] = [16, 32]
        >>> model.params['kernel_2d_size'] = [[3, 3], [3, 3]]
        >>> model.params['pool_2d_size'] = [[2, 2], [2, 2]]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='left_length', value=10,
                         desc='Length of left input.'))
        params.add(Param(name='right_length', value=100,
                         desc='Length of right input.'))
        params.add(Param(name='kernel_1d_count', value=32,
                         desc="Kernel count of 1D convolution layer."))
        params.add(Param(name='kernel_1d_size', value=3,
                         desc="Kernel size of 1D convolution layer."))
        params.add(Param(name='kernel_2d_count', value=[32],
                         desc="Kernel count of 2D convolution layer in"
                              "each block"))
        params.add(Param(name='kernel_2d_size', value=[(3, 3)],
                         desc="Kernel size of 2D convolution layer in"
                              " each block."))
        params.add(Param(name='activation', value='relu',
                         desc="Activation function."))
        params.add(Param(name='pool_2d_size', value=[(2, 2)],
                         desc="Size of pooling layer in each block."))
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

        ArcII has the desirable property of letting two sentences meet before
        their own high-level representations mature.
        """
        self.embedding = self._make_default_embedding_layer()

        # Phrase level representations
        self.conv1d_left = nn.Sequential(
            nn.ConstantPad1d((0, self._params['kernel_1d_size'] - 1), 0),
            nn.Conv1d(
                in_channels=self._params['embedding_output_dim'],
                out_channels=self._params['kernel_1d_count'],
                kernel_size=self._params['kernel_1d_size']
            )
        )
        self.conv1d_right = nn.Sequential(
            nn.ConstantPad1d((0, self._params['kernel_1d_size'] - 1), 0),
            nn.Conv1d(
                in_channels=self._params['embedding_output_dim'],
                out_channels=self._params['kernel_1d_count'],
                kernel_size=self._params['kernel_1d_size']
            )
        )

        # Interaction
        self.matching = Matching(matching_type='plus')

        # Build conv
        activation = parse_activation(self._params['activation'])
        in_channel_2d = [
            self._params['kernel_1d_count'],
            *self._params['kernel_2d_count'][:-1]
        ]
        conv2d = [
            self._make_conv_pool_block(ic, oc, ks, activation, ps)
            for ic, oc, ks, ps in zip(in_channel_2d,
                                      self._params['kernel_2d_count'],
                                      self._params['kernel_2d_size'],
                                      self._params['pool_2d_size'])
        ]
        self.conv2d = nn.Sequential(*conv2d)

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

        left_length = self._params['left_length']
        right_length = self._params['right_length']
        for ps in self._params['pool_2d_size']:
            left_length = left_length // ps[0]
        for ps in self._params['pool_2d_size']:
            right_length = right_length // ps[1]

        # Build output
        self.out = self._make_output_layer(
            left_length * right_length * self._params['kernel_2d_count'][-1]
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

        # shape = [B, L, F1]
        # shape = [B, R, F1]
        conv1d_left = self.conv1d_left(embed_left).transpose(1, 2)
        conv1d_right = self.conv1d_right(embed_right).transpose(1, 2)

        # Compute matching signal
        # shape = [B, L, R, F1]
        embed_cross = self.matching(conv1d_left, conv1d_right)

        # Convolution
        # shape = [B, F2, L // P, R // P]
        conv = self.conv2d(embed_cross.permute(0, 3, 1, 2))

        # shape = [B, F2 * (L // P) * (R // P)]
        embed_flat = self.dropout(torch.flatten(conv, start_dim=1))

        # shape = [B, *]
        out = self.out(embed_flat)
        return out

    @classmethod
    def _make_conv_pool_block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        activation: nn.Module,
        pool_size: tuple,
    ) -> nn.Module:
        """Make conv pool block."""
        return nn.Sequential(
            # Same padding
            nn.ConstantPad2d(
                (0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            activation,
            nn.MaxPool2d(kernel_size=pool_size)
        )
