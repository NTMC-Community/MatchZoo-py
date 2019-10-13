"""Matching Tensor module."""
import typing

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MatchingTensor(nn.Module):
    """
    Module that captures the basic interactions between two tensors.
    :param matching_dims: Word dimension of two interaction texts
    :param channels: Number of word interaction tensor channels
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.

    Examples:
        >>> import matchzoo as mz
        >>> left_dim, right_dim = 5, 5
        >>> matching_dims = [left_dim, right_dim]
        >>> matching_tensor = mz.modules.MatchingTensor(matching_dims,
        ...                             channels=4,
        ...                             normalize=True,
        ...                             init_diag=True)

    """
    def __init__(self, matching_dims, channels: int = 4, normalize: bool = True,
                 init_diag: bool = True):
        """:class:`MatchingTensor` constructor."""
        super().__init__()
        self._matching_dims = matching_dims
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag

        # Used purely for shape validation.
        if not isinstance(matching_dims, list) or len(matching_dims) != 2:
            raise ValueError('The parameter `matching_dims` should be a list of 2 inputs.')
            if self.matching_dims[0] != self.matching_dims[1]:
                raise ValueError(
                    'Incompatible dimensions: '
                    f'{self.matching_dims[0]} != {self.matching_dims[1]}.'
                )

        if self._init_diag:
            interaction_matrix = np.float32(
                np.random.uniform(
                    -0.05, 0.05,
                    [self._channels, self._matching_dims[0], self._matching_dims[1]]
                )
            )
            for channel_index in range(self._channels):
                np.fill_diagonal(interaction_matrix[channel_index], 0.1)
            self.interaction_matrix = nn.Parameter(torch.tensor(interaction_matrix))
        else:
            self.interaction_matrix = nn.Parameter(torch.empty(
                [self._channels, self._matching_dims[0], self._matching_dims[1]]
            ).uniform_())



    def forward(self,inputs):
        """
        The computation logic of MatchingTensor.

        :param inputs: two input tensors.
        """

        x = inputs[0]
        y = inputs[1]
        if self._normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        
        # output = [b, c, l, r]
        output = torch.einsum(
            'bld,cde,bre->bclr',
            x, self.interaction_matrix, y
        )
        return output
