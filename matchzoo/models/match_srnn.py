"""An implementation of Match-SRNN Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import MatchingTensor
from matchzoo.modules import SpatialGRU


class MatchSRNN(BaseModel):
    """
    Match-SRNN Model.

    Examples:
        >>> model = MatchSRNN()
        >>> model.params['channels'] = 4
        >>> model.params['units'] = 10
        >>> model.params['dropout'] = 0.2
        >>> model.params['direction'] = 'lt'
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
        params.add(Param(name='channels', value=4,
                         desc="Number of word interaction tensor channels"))
        params.add(Param(name='units', value=10,
                         desc="Number of SpatialGRU units"))
        params.add(Param(name='direction', value='lt',
                         desc="Direction of SpatialGRU scanning"))
        params.add(Param(
            'dropout', 0.2,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """Build model structure."""

        self.embedding = self._make_default_embedding_layer()
        self.dropout = nn.Dropout(p=self._params['dropout'])

        self.matching_tensor = MatchingTensor(
            self._params['embedding_output_dim'],
            channels=self._params["channels"])

        self.spatial_gru = SpatialGRU(
            units=self._params['units'],
            direction=self._params['direction'])

        self.out = self._make_output_layer(self._params['units'])

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   C = number of channels

        # Left input and right input
        # query = [B, L]
        # doc = [B, R]
        query, doc = inputs["text_left"].long(), inputs["text_right"].long()

        # Process left and right input
        # query = [B, L, D]
        # doc = [B, R, D]
        query = self.embedding(query)
        doc = self.embedding(doc)

        # query = [B, L, D]
        # doc = [B, R, D]
        query = self.dropout(query)
        doc = self.dropout(doc)

        # Get matching tensor
        # matching_tensor = [B, C, L, R]
        matching_tensor = self.matching_tensor(query, doc)

        # Apply spatial GRU to the word level interaction tensor
        # h_ij = [B, U]
        h_ij = self.spatial_gru(matching_tensor)

        # h_ij = [B, U]
        h_ij = self.dropout(h_ij)

        # Make output layer
        out = self.out(h_ij)

        return out
