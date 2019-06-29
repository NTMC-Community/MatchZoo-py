"""A simple densely connected baseline model."""
import typing

import torch

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class DenseBaseline(BaseModel):
    """
    A simple densely connected baseline model.

    Examples:
        >>> model = DenseBaseline()
        >>> model.params['mlp_num_layers'] = 2
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['mlp_num_fan_out'] = 128
        >>> model.params['mlp_activation_func'] = 'relu'
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
        params['mlp_num_units'] = 256
        params.get('mlp_num_units').hyper_space = \
            hyper_spaces.quniform(16, 512)
        params.get('mlp_num_layers').hyper_space = \
            hyper_spaces.quniform(1, 5)
        return params

    def build(self):
        """Build."""
        self.embeddinng = self._make_default_embedding_layer()
        self.mlp = self._make_multi_layer_perceptron_layer(
            2 * self._params['embedding_output_dim']
        )
        self.out = self._make_output_layer(
            self._params['mlp_num_fan_out']
        )

    def forward(self, inputs):
        """Forward."""
        input_left, input_right = inputs['text_left'], inputs['text_right']
        input_left = self.embeddinng(input_left.long()).sum(1)
        input_right = self.embeddinng(input_right.long()).sum(1)
        x = torch.cat((input_left, input_right), dim=1)
        return self.out(self.mlp(x))
