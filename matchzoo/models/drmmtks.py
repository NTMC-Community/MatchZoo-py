"""An implementation of DRMMTKS Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

import matchzoo
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import Attention


class DRMMTKS(BaseModel):
    """
    DRMMTKS Model.

    Examples:
        >>> model = DRMMTKS()
        >>> model.params['top_k'] = 10
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
        params.add(Param(
            'top_k', value=10,
            hyper_space=hyper_spaces.quniform(low=2, high=100),
            desc="Size of top-k pooling layer."
        ))
        params['mlp_num_fan_out'] = 1
        return params

    @classmethod
    def get_default_padding_callback(
        cls,
        fixed_length_left: int = 10,
        fixed_length_right: int = 100,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre'
    ):
        """
        Model default padding callback.

        The padding callback's on_batch_unpacked would pad a batch of data to
        a fixed length.

        :return: Default padding callback.
        """
        return matchzoo.dataloader.callbacks.BasicPadding(
            fixed_length_left=fixed_length_left,
            fixed_length_right=fixed_length_right,
            pad_value=pad_value,
            pad_mode=pad_mode)

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()
        self.attention = Attention(
            input_size=self._params['embedding_output_dim'],
            mask=self._params['mask_value']
        )
        self.mlp = self._make_multi_layer_perceptron_layer(
            self._params['top_k']
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = size of top-k

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        query, doc = inputs['text_left'], inputs['text_right']

        # Process left input.
        # shape = [B, L, D]
        embed_query = self.embedding(query.long())
        # shape = [B, R, D]
        embed_doc = self.embedding(doc.long())

        # Matching histogram of top-k
        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )

        # shape = [B, L, K]
        matching_topk = torch.topk(
            matching_matrix,
            k=self._params['top_k'],
            dim=-1,
            sorted=True
        )[0]
        # shape = [B, L]
        attention_probs = self.attention(embed_query)

        # shape = [B, L]
        dense_output = self.mlp(matching_topk).squeeze(dim=-1)

        x = torch.einsum('bl,bl->b', dense_output, attention_probs)

        out = self.out(x.unsqueeze(dim=-1))
        return out
