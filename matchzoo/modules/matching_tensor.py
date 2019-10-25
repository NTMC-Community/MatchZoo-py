"""Matching Tensor module."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingTensor(nn.Module):
    """
    Module that captures the basic interactions between two tensors.

    :param matching_dims: Word dimension of two interaction texts.
    :param channels: Number of word interaction tensor channels.
    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to True, then the output of the dot product
        is the cosine proximity between the two samples.
    :param init_diag: Whether to initialize the diagonal elements
        of the matrix.

    Examples:
        >>> import matchzoo as mz
        >>> matching_dim = 5
        >>> matching_tensor = mz.modules.MatchingTensor(
        ...    matching_dim,
        ...    channels=4,
        ...    normalize=True,
        ...    init_diag=True
        ... )

    """

    def __init__(
        self,
        matching_dim: int,
        channels: int = 4,
        normalize: bool = True,
        init_diag: bool = True
    ):
        """:class:`MatchingTensor` constructor."""
        super().__init__()
        self._matching_dim = matching_dim
        self._channels = channels
        self._normalize = normalize
        self._init_diag = init_diag

        self.interaction_matrix = torch.empty(
            self._channels, self._matching_dim, self._matching_dim
        )
        if self._init_diag:
            self.interaction_matrix = self.interaction_matrix.uniform_(-0.05, 0.05)
            for channel_index in range(self._channels):
                self.interaction_matrix[channel_index].fill_diagonal_(0.1)
            self.interaction_matrix = nn.Parameter(self.interaction_matrix)
        else:
            self.interaction_matrix = nn.Parameter(self.interaction_matrix.uniform_())

    def forward(self, x, y):
        """
        The computation logic of MatchingTensor.

        :param inputs: two input tensors.
        """

        if self._normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)

        # output = [b, c, l, r]
        output = torch.einsum(
            'bld,cde,bre->bclr',
            x, self.interaction_matrix, y
        )
        return output
