"""An implementation of RE2 Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import (
    Encoder, Alignment, FullFusion, AugmentedResidual, Pooling
)


class RE2(BaseModel):
    """
    RE2 Model.

    Examples:
        >>> model = RE2()
        >>> model.params['mask_value'] = 0
        >>> model.params['num_blocks'] = 3
        >>> model.params['num_encoder_layers'] = 3
        >>> model.params['kernel_size'] = 3
        >>> model.params['hidden_size'] = 200
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
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='num_blocks', value=2,
                         desc="Number of blocks."))
        params.add(Param(name='num_encoder_layers', value=3,
                         desc="Number of encoder layers."))
        params.add(Param(name='kernel_size', value=3,
                         desc="Kernel size of 1D convolution layer."))
        params.add(Param(name='hidden_size', value=200,
                         desc="Hidden size."))
        params.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="Float, the dropout rate."
        ))
        return params

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()

        embedding_dim = self._params['embedding_output_dim']
        hidden_size = self._params['hidden_size']

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'encoder': Encoder(
                    num_encoder_layers=self._params['num_encoder_layers'],
                    input_size=embedding_dim if i == 0 else embedding_dim + hidden_size,
                    hidden_size=hidden_size,
                    kernel_size=self._params['kernel_size'],
                    dropout_rate=self._params['dropout_rate']
                ),
                'alignment': Alignment(
                    hidden_size=embedding_dim + hidden_size if i == 0
                    else embedding_dim + hidden_size * 2),
                'fusion': FullFusion(
                    input_size=embedding_dim + hidden_size if i == 0
                    else embedding_dim + hidden_size * 2,
                    hidden_size=hidden_size,
                    dropout=self._params['dropout_rate']
                )
            }) for i in range(self._params['num_blocks'])]
        )
        self.connection = AugmentedResidual()
        self.pooling = Pooling()
        self.out = self._make_output_layer(hidden_size * 2)

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = hidden size

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # shape = [B, L]
        # shape = [B, R]
        mask_left = (input_left == self._params['mask_value'])
        mask_right = (input_right == self._params['mask_value'])

        # Process left input.
        # shape = [B, L, D]
        # shape = [B, R, D]
        left = self.embedding(input_left.long())
        right = self.embedding(input_right.long())
        res_left, res_right = left, right

        for i, block in enumerate(self.blocks):
            if i > 0:
                # shape = [B, L, H + D]
                # shape = [B, R, H + D]
                left = self.connection(left, res_left, i)
                right = self.connection(right, res_right, i)

                res_left, res_right = left, right

            # shape = [B, L, H]
            # shape = [B, R, H]
            enc_left = block['encoder'](left, mask_left)
            enc_right = block['encoder'](right, mask_right)

            # shape = [B, L, 2 * H + D]
            # shape = [B, L, 2 * H + D]
            left = torch.cat([left, enc_left], dim=-1)
            right = torch.cat([right, enc_right], dim=-1)

            # shape = [B, L, 2 * H + D]
            # shape = [B, L, 2 * H + D]
            align_left, align_right = block['alignment'](
                left, right, mask_left, mask_right
            )

            # shape = [B, L, H]
            # shape = [B, R, H]
            left = block['fusion'](left, align_left)
            right = block['fusion'](right, align_right)

        # shape = [B, H]
        # shape = [B, H]
        left = self.pooling(left, mask_left)
        right = self.pooling(right, mask_right)

        # shape = [B, *]
        out = self.out(torch.cat([left, right], dim=-1))

        return out
