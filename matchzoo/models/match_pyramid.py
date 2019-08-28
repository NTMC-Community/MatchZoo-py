"""An implementation of MatchPyramid Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import Matching
from matchzoo.dataloader import callbacks
from matchzoo.utils import parse_activation


class MatchPyramid(BaseModel):
    """
    MatchPyramid Model.

    Examples:
        >>> model = MatchPyramid()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['kernel_count'] = [16, 32]
        >>> model.params['kernel_size'] = [[3, 3], [3, 3]]
        >>> model.params['dpool_size'] = [3, 10]
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='kernel_count', value=[32],
                         desc="The kernel count of the 2D convolution "
                              "of each block."))
        params.add(Param(name='kernel_size', value=[[3, 3]],
                         desc="The kernel size of the 2D convolution "
                              "of each block."))
        params.add(Param(name='activation', value='relu',
                         desc="The activation function."))
        params.add(Param(name='dpool_size', value=[3, 10],
                         desc="The max-pooling size of each block."))

        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        MatchPyramid text matching as image recognition.
        """
        self.embedding = self._make_default_embedding_layer()

        # Interaction
        self.matching = Matching(matching_type='dot')

        # Build conv
        activation = parse_activation(self._params['activation'])
        in_channel_2d = [
            1,
            *self._params['kernel_count'][:-1]
        ]
        conv2d = [
            self._make_conv_pool_block(ic, oc, ks, activation)
            for ic, oc, ks, in zip(in_channel_2d,
                                   self._params['kernel_count'],
                                   self._params['kernel_size'])
        ]
        self.conv2d = nn.Sequential(*conv2d)

        # Dynamic Pooling
        self.dpool_layer = nn.AdaptiveAvgPool2d(self._params['dpool_size'])

        self.dropout = nn.Dropout(p=self._params['dropout_rate'])

        left_length = self._params['dpool_size'][0]
        right_length = self._params['dpool_size'][1]

        # Build output
        self.out = self._make_output_layer(
            left_length * right_length * self._params['kernel_count'][-1]
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
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        # Compute matching signal
        # shape = [B, 1, L, R]
        embed_cross = self.matching(embed_left, embed_right).unsqueeze(dim=1)

        # Convolution
        # shape = [B, F, L, R]
        conv = self.conv2d(embed_cross)

        # Dynamic Pooling
        # shape = [B, F, P1, P2]
        embed_pool = self.dpool_layer(conv)

        # shape = [B, F * P1 * P2]
        embed_flat = self.dropout(torch.flatten(embed_pool, start_dim=1))

        # shape = [B, *]
        out = self.out(embed_flat)
        return out

    @classmethod
    def _make_conv_pool_block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        activation: nn.Module
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
            activation
        )
