"""An implementation of Dynamic Clip Model."""
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
from matchzoo.utils import parse_activation
from matchzoo.modules import DynamicClipAttention


class DynamicClip(BaseModel):
    """
    Dynamic Clip Model.

    Examples:
        >>> model = DynamicClip()
        >>> model.params['clip_type'] = 'max'
        >>> model.params['left_length'] = 10
        >>> model.params['right_length'] = 100
        >>> model.params['hidden_dim'] = 128
        >>> model.params['kernel_size'] = 3
        >>> model.params['pool_size'] = 2
        >>> model.params['left_topk'] = 10
        >>> model.params['right_topk'] = 5
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
        params.add(Param(name='left_length', value=10,
                         desc='Length of left input.'))
        params.add(Param(name='right_length', value=100,
                         desc='Length of right input.'))
        params.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        params.add(Param(name='hidden_activation', value='tanh',
                         desc="The activation function in the "
                         "hidden layer."))
        params.add(Param(name='conv_activation', value='relu',
                         desc="The activation function in the "
                         "convolution layer."))
        params.add(Param(name='hidden_dim', value=128,
                         desc="Number of hidden dimensions."))
        params.add(Param(name='kernel_size', value=3,
                         desc="Number of kernel size in the 1D "
                         "convolution layer."))
        params.add(Param(name='pool_size', value=2,
                         desc="The pooling size of convolution."))
        params.add(Param(name='clip_type', value='max',
                         desc="The type of dynamic clip attention."))
        params.add(Param(name='left_topk', value=10,
                         desc="Keep topk weights in the"
                         "attention block for the left input."))
        params.add(Param(name='right_topk', value=5,
                         desc="Keep topk weights in the"
                         "attention block for the right input."))
        params.add(Param(name='left_threshold', value=0.08,
                         desc="Keep weights above the threshold"
                         "in the attention block for the left input."))
        params.add(Param(name='right_threshold', value=0.08,
                         desc="Keep weights above the threshold"
                         "in the attention block for the right input."))
        params.add(Param(
            'dropout', 0.2,
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
            pad_word_mode=pad_word_mode
        )

    def build(self):
        """Build model structure."""

        self.embedding = self._make_default_embedding_layer()
        self.dropout = nn.Dropout(p=self._params['dropout'])
        self.attention = DynamicClipAttention(
            clip_type=self._params['clip_type'],
            topk=(self._params['left_topk'], self._params['right_topk']),
            threshold=(self._params['left_threshold'], self._params['right_threshold']))
        self.conv_activation = parse_activation(self._params['conv_activation'])
        self.hidden_activation = parse_activation(self._params['hidden_activation'])
        self.conv1d_left = nn.Sequential(
            nn.ConstantPad1d((0, self._params['kernel_size'] - 1), 0),
            nn.Conv1d(
                in_channels=self._params['embedding_output_dim'],
                out_channels=self._params['hidden_dim'],
                kernel_size=self._params['kernel_size']
            ),
            self.conv_activation,
            nn.MaxPool1d(kernel_size=self._params['pool_size']),
        )
        self.conv1d_right = nn.Sequential(
            nn.ConstantPad1d((0, self._params['kernel_size'] - 1), 0),
            nn.Conv1d(
                in_channels=self._params['embedding_output_dim'],
                out_channels=self._params['hidden_dim'],
                kernel_size=self._params['kernel_size']
            ),
            self.conv_activation,
            nn.MaxPool1d(kernel_size=self._params['pool_size']),
        )
        ps = self._params['pool_size']
        left_length = self._params['left_length']
        right_length = self._params['right_length']
        concat_length = left_length // ps + right_length // ps
        self.feature_to_dense = nn.Linear(concat_length * self._params['hidden_dim'],
                           self._params['hidden_dim'])
        self.out = self._make_output_layer(self._params['hidden_dim'])

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = number of dimensions in hidden layer
        #   P = pool size of convolution

        # Left input and right input.
        # input_left = [B, L]
        # input_right = [B, R]
        input_left, input_right = inputs["text_left"].long(), inputs["text_right"].long()

        # left_mask = [B, L]
        # right_mask = [B, R]
        left_mask = (input_left == self._params['mask_value'])
        right_mask = (input_right == self._params['mask_value'])

        # Process left and right input.
        # embed_left = [B, L, D]
        # embed_right = [B, R, D]
        embed_left = self.embedding(input_left)
        embed_right = self.embedding(input_right)

        # embed_left = [B, L, D]
        # embed_right = [B, R, D]
        embed_left = self.dropout(embed_left)
        embed_right = self.dropout(embed_right)

        # Get attention.
        # attended_left = [B, L, D]
        # attention_right = [B, R, D]
        attended_left, attended_right = self.attention(
            embed_left, left_mask, embed_right, right_mask)

        # Element-wise comparison with origin sequence.
        # cmp_left = [B, L, D]
        # cmp_right = [B, R, D]
        cmp_left = embed_left * attended_left
        cmp_right = embed_right * attended_right

        # Process left and right input.
        # cmp_left = [B, D, L]
        # cmp_right = [B, D, R]
        cmp_left = self.hidden_activation(cmp_left).transpose(1, 2)
        cmp_right = self.hidden_activation(cmp_right).transpose(1, 2)

        # Aggregate by convolution.
        # agg_left = [B, H, L // P]
        # agg_right = [B, H, R // P]
        agg_left = self.conv1d_left(cmp_left)
        agg_right = self.conv1d_right(cmp_right)

        # Concat left and right.
        # concat = [B, H, (L // P + R // P)]
        # rep_concat = [B, H * (L // P+ R // P)]
        concat = torch.cat([agg_left, agg_right], dim=2)
        rep_concat = torch.flatten(concat, start_dim=1)

        # Score.
        score = self.feature_to_dense(rep_concat)

        # Make output layer.
        out = self.out(score)

        return out
