"""An implementation of DSSM, Deep Structured Semantic Model."""
import torch
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo import preprocessors


class DSSM(BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model = DSSM()
        >>> model.params['mlp_num_layers'] = 3
        >>> model.params['mlp_num_units'] = 300
        >>> model.params['mlp_num_fan_out'] = 128
        >>> model.params['mlp_activation_func'] = 'relu'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params.add(Param(name='vocab_size', value=4,
                         desc="Size of vocabulary."))
        return params

    def build(self):
        """
        Build model structure.

        DSSM use Siamese arthitecture.
        """
        self.mlp_left = self._make_multi_layer_perceptron_layer(
            self._params['vocab_size']
        )
        self.mlp_right = self._make_multi_layer_perceptron_layer(
            self._params['vocab_size']
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Process left & right input.
        input_left, input_right = inputs['text_left'], inputs['text_right']
        input_left = self.mlp_left(input_left)
        input_right = self.mlp_right(input_right)

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)

        out = self.out(x.unsqueeze(dim=1))
        return out
