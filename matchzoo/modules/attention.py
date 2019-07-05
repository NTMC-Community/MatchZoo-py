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
        x = self.linear(x).squeeze(dim=-1)
        mask = (x != self.mask)
        x = x.masked_fill(mask == self.mask, -float('inf'))
        return F.softmax(x, dim=-1)


class BidirectionalAttention(nn.Module):
    """
    Computing the soft attention between two sequence.
    """
    def __init__(self):
        super(BidirectionalAttention, self).__init__()

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward. """
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        v2_v1_attn = F.softmax(similarity_matrix.masked_fill(v1_mask.unsqueeze(2), -1e-7), dim=1)
        v1_v2_attn = F.softmax(similarity_matrix.masked_fill(v2_mask.unsqueeze(1), -1e-7), dim=2)

        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2
