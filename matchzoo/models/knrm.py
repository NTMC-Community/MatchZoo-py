"""An implementation of KNRM Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine import hyper_spaces
from matchzoo.modules import GaussianKernel


class KNRM(BaseModel):
    """
    KNRM Model.

    Examples:
        >>> model = KNRM()
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

        self.kernels = nn.ModuleList()
        for i in range(self._params['kernel_num']):
            mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
                self._params['kernel_num'] - 1) - 1.0
            sigma = self._params['sigma']
            if mu > 1.0:
                sigma = self._params['exact_sigma']
                mu = 1.0
            self.kernels.append(GaussianKernel(mu=mu, sigma=sigma))

        self.out = self._make_output_layer(self._params['kernel_num'])

    def forward(self, inputs):
        """Forward."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   K = number of kernels

        # Left input and right input.
        # shape = [B, L]
        # shape = [B, R]
        query, doc = inputs['text_left'], inputs['text_right']

        # Process left input.
        # shape = [B, L, D]
        embed_query = self.embedding(query.long())
        # shape = [B, R, D]
        embed_doc = self.embedding(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )

        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        phi = torch.stack(KM, dim=1)

        out = self.out(phi)
        return out
