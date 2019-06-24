"""Base Model."""

import abc
import typing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matchzoo
from matchzoo.utils import parse_activation
from matchzoo.dataloader.callbacks import Callback
from matchzoo.engine import hyper_spaces
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo import tasks


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base class of all MatchZoo models.

    MatchZoo models are wrapped over pytorch models. `params` is a set of model
    hyper-parameters that deterministically builds a model. In other words,
    `params['model_class'](params=params)` of the same `params` always create
    models with the same structure.

    :param params: Model hyper-parameters. (default: return value from
        :meth:`get_default_params`)

    Example:
        >>> BaseModel()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        TypeError: Can't instantiate abstract class BaseModel ...
        >>> class MyModel(BaseModel):
        ...     def build(self):
        ...         pass
        ...     def forward(self):
        ...         pass
        >>> isinstance(MyModel(), BaseModel)
        True

    """

    def __init__(
        self,
        params: typing.Optional[ParamTable] = None
    ):
        """Init."""
        super().__init__()
        self._params = params or self.get_default_params()

    @classmethod
    def get_default_params(
        cls,
        with_embedding=False,
        with_multi_layer_perceptron=False
    ) -> ParamTable:
        """
        Model default parameters.

        The common usage is to instantiate :class:`matchzoo.engine.ModelParams`
            first, then set the model specific parametrs.

        Examples:
            >>> class MyModel(BaseModel):
            ...     def build(self):
            ...         print(self._params['num_eggs'], 'eggs')
            ...         print('and', self._params['ham_type'])
            ...     def forward(self, greeting):
            ...         print(greeting)
            ...
            ...     @classmethod
            ...     def get_default_params(cls):
            ...         params = ParamTable()
            ...         params.add(Param('num_eggs', 512))
            ...         params.add(Param('ham_type', 'Parma Ham'))
            ...         return params
            >>> my_model = MyModel()
            >>> my_model.build()
            512 eggs
            and Parma Ham
            >>> my_model('Hello MatchZoo!')
            Hello MatchZoo!

        Notice that all parameters must be serialisable for the entire model
        to be serialisable. Therefore, it's strongly recommended to use python
        native data types to store parameters.

        :return: model parameters

        """
        params = ParamTable()
        params.add(Param(
            name='model_class', value=cls,
            desc="Model class. Used internally for save/load. "
                 "Changing this may cause unexpected behaviors."
        ))
        params.add(Param(
            name='task',
            desc="Decides model output shape, loss, and metrics."
        ))
        if with_embedding:
            params.add(Param(
                name='with_embedding', value=True,
                desc="A flag used help `auto` module. Shouldn't be changed."
            ))
            params.add(Param(
                name='embedding',
                desc='FloatTensor containing weights for the Embedding.',
                validator=lambda x: isinstance(x, np.ndarray)
            ))
            params.add(Param(
                name='embedding_input_dim',
                desc='Usually equals vocab size + 1. Should be set manually.'
            ))
            params.add(Param(
                name='embedding_output_dim',
                desc='Should be set manually.'
            ))
            params.add(Param(
                name='embedding_trainable', value=True,
                desc='`True` to enable embedding layer training, '
                     '`False` to freeze embedding parameters.'
            ))
        if with_multi_layer_perceptron:
            params.add(Param(
                name='with_multi_layer_perceptron', value=True,
                desc="A flag of whether a multiple layer perceptron is used. "
                     "Shouldn't be changed."
            ))
            params.add(Param(
                name='mlp_num_units', value=128,
                desc="Number of units in first `mlp_num_layers` layers.",
                hyper_space=hyper_spaces.quniform(8, 256, 8)
            ))
            params.add(Param(
                name='mlp_num_layers', value=3,
                desc="Number of layers of the multiple layer percetron.",
                hyper_space=hyper_spaces.quniform(1, 6)
            ))
            params.add(Param(
                name='mlp_num_fan_out', value=64,
                desc="Number of units of the layer that connects the multiple "
                     "layer percetron and the output.",
                hyper_space=hyper_spaces.quniform(4, 128, 4)
            ))
            params.add(Param(
                name='mlp_activation_func', value='relu',
                desc='Activation function used in the multiple '
                     'layer perceptron.'
            ))
        return params

    def guess_and_fill_missing_params(self, verbose=1):
        """
        Guess and fill missing parameters in :attr:`params`.

        Use this method to automatically fill-in other hyper parameters.
        This involves some guessing so the parameter it fills could be
        wrong. For example, the default task is `Ranking`, and if we do not
        set it to `Classification` manaully for data packs prepared for
        classification, then the shape of the model output and the data will
        mismatch.

        :param verbose: Verbosity.
        """
        self._params.get('task').set_default(tasks.Ranking(), verbose)
        if 'with_embedding' in self._params:
            self._params.get('embedding_input_dim').set_default(300, verbose)
            self._params.get('embedding_output_dim').set_default(300, verbose)

    def _set_param_default(self, name: str,
                           default_val: str, verbose: int = 0):
        if self._params[name] is None:
            self._params[name] = default_val
            if verbose:
                print(f"Parameter \"{name}\" set to {default_val}.")

    @classmethod
    def get_default_preprocessor(cls) -> BasePreprocessor:
        """
        Model default preprocessor.

        The preprocessor's transform should produce a correctly shaped data
        pack that can be used for training.

        :return: Default preprocessor.
        """
        return matchzoo.preprocessors.BasicPreprocessor()

    @classmethod
    def get_default_padding_callback(cls) -> Callback:
        """
        Model default padding callback.

        The padding callback's on_batch_unpacked would pad a batch of data to
        fixed length.

        :return: Default padding callback.
        """
        return matchzoo.dataloader.callbacks.BasicPadding()

    @property
    def params(self) -> ParamTable:
        """:return: model parameters."""
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

    @abc.abstractmethod
    def build(self):
        """Build model, each subclass need to implement this method."""
        raise NotImplementedError(
            "Build method not implemented in the subclass."
        )

    @abc.abstractmethod
    def forward(self, *input):
        """
        Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(
            "Forward method not implemented in the subclass."
        )

    def _make_embedding_layer(
        self,
        num_embeddings: int = 0,
        embedding_dim: int = 0,
        freeze: bool = True,
        embedding: typing.Optional[np.ndarray] = None,
        **kwargs
    ) -> nn.Module:
        """:return: an embedding module."""
        if not embedding and isinstance(
            self._params['embedding'], np.ndarray
        ):
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(embedding),
                freeze=freeze
            )
        else:
            return nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim
            )

    def _make_default_embedding_layer(
        self,
        **kwargs
    ) -> nn.Module:
        """:return: an embedding module."""
        if isinstance(self._params['embedding'], np.ndarray):
            self._params['embedding_input_dim'] = (
                self._params['embedding'].shape[0]
            )
            self._params['embedding_output_dim'] = (
                self._params['embedding'].shape[1]
            )
            return nn.Embedding.from_pretrained(
                embeddings=torch.Tensor(self._params['embedding']),
                freeze=self._params['embedding_trainable']
            )
        else:
            return nn.Embedding(
                num_embeddings=self._params['embedding_input_dim'],
                embedding_dim=self._params['embedding_output_dim']
            )

    def _make_output_layer(self, in_features) -> nn.Module:
        """:return: a correctly shaped torch module for model output."""
        task = self._params['task']
        if isinstance(task, tasks.Classification):
            return nn.Sequential(
                nn.Linear(in_features, task.num_classes),
                nn.Softmax()
            )
        elif isinstance(task, tasks.Ranking):
            return nn.Sequential(
                nn.Linear(in_features, 1)
            )
        else:
            raise ValueError(f"{task} is not a valid task type. "
                             f"Must be in `Ranking` and `Classification`.")

    def _make_perceptron_layer(
        self,
        in_features: int = 0,
        out_features: int = 0,
        activation: nn.Module = nn.ReLU
    ) -> nn.Module:
        """:return: a perceptron layer."""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            activation
        )

    def _make_multi_layer_perceptron_layer(self, in_features) -> nn.Module:
        """:return: a multiple layer perceptron."""
        if not self._params['with_multi_layer_perceptron']:
            raise AttributeError(
                'Parameter `with_multi_layer_perception` not set.')

        activation = parse_activation(self._params['mlp_activation_func'])
        mlp_sizes = [
            in_features,
            *self._params['mlp_num_layers'] * [self._params['mlp_num_units']],
            self._params['mlp_num_fan_out']
        ]
        mlp = [
            self._make_perceptron_layer(in_f, out_f, activation)
            for in_f, out_f in zip(mlp_sizes, mlp_sizes[1:])
        ]
        return nn.Sequential(*mlp)
