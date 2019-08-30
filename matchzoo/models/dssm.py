"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
import torch.nn.functional as F

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor


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
        params.add(Param(name='vocab_size', value=419,
                         desc="Size of vocabulary."))
        return params

    @classmethod
    def get_default_preprocessor(
        cls,
        truncated_mode: str = 'pre',
        truncated_length_left: typing.Optional[int] = None,
        truncated_length_right: typing.Optional[int] = None,
        filter_mode: str = 'df',
        filter_low_freq: float = 1,
        filter_high_freq: float = float('inf'),
        remove_stop_words: bool = False,
        ngram_size: typing.Optional[int] = 3,
    ) -> BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training.

        :return: Default preprocessor.
        """
        return preprocessors.BasicPreprocessor(
            truncated_mode=truncated_mode,
            truncated_length_left=truncated_length_left,
            truncated_length_right=truncated_length_right,
            filter_mode=filter_mode,
            filter_low_freq=filter_low_freq,
            filter_high_freq=filter_high_freq,
            remove_stop_words=remove_stop_words,
            ngram_size=ngram_size
        )

    @classmethod
    def get_default_padding_callback(cls):
        """:return: Default padding callback."""
        return None

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
        input_left, input_right = inputs['ngram_left'], inputs['ngram_right']
        input_left = self.mlp_left(input_left)
        input_right = self.mlp_right(input_right)

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)

        out = self.out(x.unsqueeze(dim=1))
        return out
