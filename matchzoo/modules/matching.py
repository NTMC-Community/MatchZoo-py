"""Matching module."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class Matching(nn.Module):
    """
    Module that computes a matching matrix between samples in two tensors.

    :param normalize: Whether to L2-normalize samples along the
        dot product axis before taking the dot product.
        If set to `True`, then the output of the dot product
        is the cosine proximity between the two samples.
    :param matching_type: the similarity function for matching

    Examples:
        >>> import torch
        >>> matching = Matching(matching_type='dot', normalize=True)
        >>> x = torch.randn(2, 3, 2)
        >>> y = torch.randn(2, 4, 2)
        >>> matching(x, y).shape
        torch.Size([2, 3, 4])

    """

    def __init__(self, normalize: bool = False, matching_type: str = 'dot'):
        """:class:`Matching` constructor."""
        super().__init__()
        self._normalize = normalize
        self._validate_matching_type(matching_type)
        self._matching_type = matching_type

    @classmethod
    def _validate_matching_type(cls, matching_type: str = 'dot'):
        valid_matching_type = ['dot', 'exact', 'mul', 'plus', 'minus', 'concat']
        if matching_type not in valid_matching_type:
            raise ValueError(f"{matching_type} is not a valid matching type, "
                             f"{valid_matching_type} expected.")

    def forward(self, x, y):
        """Perform attention on the input."""
        length_left = x.shape[1]
        length_right = y.shape[1]
        if self._matching_type == 'dot':
            if self._normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)
            return torch.einsum('bld,brd->blr', x, y)
        elif self._matching_type == 'exact':
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1)
            matching_matrix = (x == y)
            x = torch.sum(matching_matrix, dim=2, dtype=torch.float)
            y = torch.sum(matching_matrix, dim=1, dtype=torch.float)
            return x, y
        else:
            x = x.unsqueeze(dim=2).repeat(1, 1, length_right, 1)
            y = y.unsqueeze(dim=1).repeat(1, length_left, 1, 1)
            if self._matching_type == 'mul':
                return x * y
            elif self._matching_type == 'plus':
                return x + y
            elif self._matching_type == 'minus':
                return x - y
            elif self._matching_type == 'concat':
                return torch.cat((x, y), dim=3)
