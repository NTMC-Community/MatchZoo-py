"""Attention module."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention module.

    :param input_size: Size of input.
    :param mask: An integer to mask the invalid values. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = Attention(input_size=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> attention(x).shape
        torch.Size([4, 5])

    """

    def __init__(self, input_size: int = 100, mask: int = 0):
        """Attention constructor."""
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)
        self.mask = mask

    def forward(self, x):
        """Perform attention on the input."""
        x = self.linear(x).squeeze()
        mask = (x != self.mask)
        x = x.masked_fill(mask == self.mask, -float('inf'))
        return F.softmax(x, dim=-1)
