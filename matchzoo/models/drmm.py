"""An implementation of DRMM Model."""
import typing

import torch
import torch.nn as nn

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.dataloader import callbacks
from matchzoo.modules import Attention


class DRMM(BaseModel):
    """
    DRMM Model.

    Examples:
        >>> model = DRMM()
        >>> model.params['mlp_num_layers'] = 1
        >>> model.params['mlp_num_units'] = 5
        >>> model.params['mlp_num_fan_out'] = 1
        >>> model.params['mlp_activation_func'] = 'tanh'
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
        params.add(Param(name='hist_bin_size', value=30,
                         desc="The number of bin size of the histogram."))
        params['mlp_num_fan_out'] = 1
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre'
    ):
        """:return: Default padding callback."""
        return callbacks.DRMMPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_value=pad_value,
            pad_mode=pad_mode
        )

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()
        self.attention = Attention(
            input_size=self._params['embedding_output_dim']
        )
        self.mlp = self._make_multi_layer_perceptron_layer(
            self._params['hist_bin_size']
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   H = histogram size
        #   K = size of top-k

        # Left input and right input.
        # query: shape = [B, L]
        # doc: shape = [B, L, H]
        # Note here, the doc is the matching histogram between original query
        # and original document.

        query, match_hist = inputs['text_left'], inputs['match_histogram']

        # shape = [B, L]
        mask_query = (query == self._params['mask_value'])

        # Process left input.
        # shape = [B, L, D]
        embed_query = self.embedding(query.long())

        # shape = [B, L]
        attention_probs = self.attention(embed_query, mask_query)

        # shape = [B, L]
        dense_output = self.mlp(match_hist).squeeze(dim=-1)

        x = torch.einsum('bl,bl->b', dense_output, attention_probs)

        out = self.out(x.unsqueeze(dim=-1))
        return out
