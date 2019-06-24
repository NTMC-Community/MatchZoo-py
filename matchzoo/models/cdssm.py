"""An implementation of CDSSM (CLSM) model."""
import typing

import torch
from torch import nn
import torch.nn.functional as F

from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.dataloader import callbacks
from matchzoo import preprocessors
from matchzoo.utils import TensorType, parse_activation


class CDSSM(BaseModel):
    """
    CDSSM Model implementation.

    Learning Semantic Representations Using Convolutional Neural Networks
    for Web Search. (2014a)
    A Latent Semantic Model with Convolutional-Pooling Structure for
    Information Retrieval. (2014b)

    Examples:
        >>> import matchzoo as mz
        >>> model = CDSSM()
        >>> model.params['task'] = mz.tasks.Ranking()
        >>> model.params['vocab_size'] = 4
        >>> model.params['filters'] =  32
        >>> model.params['kernel_size'] = 3
        >>> model.params['conv_activation_func'] = 'relu'
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        # set :attr:`with_multi_layer_perceptron` to False to support
        # user-defined variable dense layer units
        params = super().get_default_params(with_multi_layer_perceptron=True)
        params.add(Param(name='vocab_size', value=4,
                         desc="Size of vocabulary."))
        params.add(Param(name='filters', value=4,
                         desc="Number of filters in the 1D convolution "
                              "layer."))
        params.add(Param(name='kernel_size', value=3,
                         desc="Number of kernel size in the 1D "
                              "convolution layer."))
        params.add(Param(name='strides', value=1,
                         desc="Strides in the 1D convolution layer."))
        params.add(Param(name='padding', value=0,
                         desc="The padding mode in the convolution "
                              "layer. It should be one of `same`, "
                              "`valid`, ""and `causal`."))
        params.add(Param(name='conv_activation_func', value='relu',
                         desc="Activation function in the convolution"
                              " layer."))
        params.add(Param(name='w_initializer', value='glorot_normal'))
        params.add(Param(name='b_initializer', value='zeros'))
        params.add(Param(name='dropout_rate', value=0.3,
                         desc="The dropout rate."))
        return params

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.CDSSMPreprocessor()

    @classmethod
    def get_default_padding_callback(cls):
        """:return: Default padding callback."""
        return callbacks.CDSSMPadding()

    def _create_base_network(self) -> nn.Module:
        """
        Apply conv and maxpooling operation towards to each letter-ngram.

        The input shape is `fixed_text_length`*`number of letter-ngram`,
        as described in the paper, `n` is 3, `number of letter-trigram`
        is about 30,000 according to their observation.

        :return: A :class:`nn.Module` of CDSSM network, tensor in tensor out.
        """
        conv = nn.Conv1d(
            in_channels=self._params['vocab_size'],
            out_channels=self._params['filters'],
            kernel_size=self._params['kernel_size'],
            stride=self._params['strides'],
            padding=self._params['padding']
        )
        activation = parse_activation(
            self._params['conv_activation_func']
        )
        dropout = nn.Dropout(p=self._params['dropout_rate'])
        pool = nn.AdaptiveMaxPool1d(1)
        squeeze = Squeeze()
        mlp = self._make_multi_layer_perceptron_layer(
            self._params['filters']
        )
        return nn.Sequential(
            conv, activation, dropout, pool, squeeze, mlp
        )

    def build(self):
        """
        Build model structure.

        CDSSM use Siamese architecture.
        """
        self.net_left = self._create_base_network()
        self.net_right = self._create_base_network()
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Process left & right input.
        input_left, input_right = inputs['text_left'], inputs['text_right']
        input_left = input_left.transpose(1, 2)
        input_right = input_right.transpose(1, 2)
        input_left = self.net_left(input_left)
        input_right = self.net_right(input_right)

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)

        out = self.out(x.unsqueeze(dim=1))
        return out

    def guess_and_fill_missing_params(self, verbose: int = 1):
        """
        Guess and fill missing parameters in :attr:`params`.

        Use this method to automatically fill-in hyper parameters.
        This involves some guessing so the parameter it fills could be
        wrong. For example, the default task is `Ranking`, and if we do not
        set it to `Classification` manually for data packs prepared for
        classification, then the shape of the model output and the data will
        mismatch.

        :param verbose: Verbosity.
        """
        super().guess_and_fill_missing_params(verbose)   


class Squeeze(nn.Module):
    """Squeeze."""

    def forward(self, x):
        """Forward."""
        return x.squeeze()
