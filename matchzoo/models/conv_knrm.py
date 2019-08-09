"""An implementation of ConvKNRM Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import GaussianKernel
from matchzoo.utils import parse_activation


class ConvKNRM(BaseModel):
    """
    ConvKNRM Model.

    Examples:
        >>> model = ConvKNRM()
        >>> model.params['filters'] = 128
        >>> model.params['conv_activation_func'] = 'tanh'
        >>> model.params['max_ngram'] = 3
        >>> model.params['use_crossmatch'] = True
        >>> model.params['kernel_num'] = 11
        >>> model.params['sigma'] = 0.1
        >>> model.params['exact_sigma'] = 0.001
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(
            name='filters',
            value=128,
            desc="The filter size in the convolution layer."
        ))
        params.add(Param(
            name='conv_activation_func',
            value='relu',
            desc="The activation function in the convolution layer."))
        params.add(Param(
            name='max_ngram',
            value=3,
            desc="The maximum length of n-grams for the convolution "
                 "layer."))
        params.add(Param(
            name='use_crossmatch',
            value=True,
            desc="Whether to match left n-grams and right n-grams of "
                 "different lengths"))
        params.add(Param(
            name='kernel_num',
            value=11,
            hyper_space=hyper_spaces.quniform(low=5, high=20),
            desc="The number of RBF kernels."
        ))
        params.add(Param(
            name='sigma',
            value=0.1,
            hyper_space=hyper_spaces.quniform(
                low=0.01, high=0.2, q=0.01),
            desc="The `sigma` defines the kernel width."
        ))
        params.add(Param(
            name='exact_sigma', value=0.001,
            desc="The `exact_sigma` denotes the `sigma` "
                 "for exact match."
        ))
        return params

    def build(self):
        """Build model structure."""
        self.embedding = self._make_default_embedding_layer()

        self.q_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
        for i in range(self._params['max_ngram']):
            conv = nn.Sequential(
                nn.ConstantPad1d((0, i), 0),
                nn.Conv1d(
                    in_channels=self._params['embedding_output_dim'],
                    out_channels=self._params['filters'],
                    kernel_size=i + 1
                ),
                parse_activation(
                    self._params['conv_activation_func']
                )
            )
            self.q_convs.append(conv)
            self.d_convs.append(conv)

        self.kernels = nn.ModuleList()
        for i in range(self._params['kernel_num']):
            mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
                self._params['kernel_num'] - 1) - 1.0
            sigma = self._params['sigma']
            if mu > 1.0:
                sigma = self._params['exact_sigma']
                mu = 1.0
            self.kernels.append(GaussianKernel(mu=mu, sigma=sigma))

        dim = self._params['max_ngram'] ** 2 * self._params['kernel_num']
        self.out = self._make_output_layer(dim)

    def forward(self, inputs):
        """Forward."""

        query, doc = inputs['text_left'], inputs['text_right']

        q_embed = self.embedding(query.long()).transpose(1, 2)
        d_embed = self.embedding(doc.long()).transpose(1, 2)

        q_convs = []
        d_convs = []
        for q_conv, d_conv in zip(self.q_convs, self.d_convs):
            q_convs.append(q_conv(q_embed).transpose(1, 2))
            d_convs.append(d_conv(d_embed).transpose(1, 2))

        KM = []
        for qi in range(self._params['max_ngram']):
            for di in range(self._params['max_ngram']):
                # do not match n-gram with different length if use crossmatch
                if not self._params['use_crossmatch'] and qi != di:
                    continue
                mm = torch.einsum(
                    'bld,brd->blr',
                    F.normalize(q_convs[qi], p=2, dim=-1),
                    F.normalize(d_convs[di], p=2, dim=-1)
                )
                for kernel in self.kernels:
                    K = torch.log1p(kernel(mm).sum(dim=-1)).sum(dim=-1)
                    KM.append(K)

        phi = torch.stack(KM, dim=1)

        out = self.out(phi)
        return out
