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
    """Computing the soft attention between two sequence."""

    def __init__(self):
        """Init."""
        super().__init__()

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward."""
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        v2_v1_attn = F.softmax(
            similarity_matrix.masked_fill(
                v1_mask.unsqueeze(2), -1e-7), dim=1)
        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1), -1e-7), dim=2)

        attended_v1 = v1_v2_attn.bmm(v2)
        attended_v2 = v2_v1_attn.transpose(1, 2).bmm(v1)

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2


class MatchModule(nn.Module):
    """
    Computing the match representation for Match LSTM.

    :param hidden_size: Size of hidden vectors.
    :param dropout_rate: Dropout rate of the projection layer. Defaults to 0.

    Examples:
        >>> import torch
        >>> attention = MatchModule(hidden_size=10)
        >>> v1 = torch.randn(4, 5, 10)
        >>> v1.shape
        torch.Size([4, 5, 10])
        >>> v2 = torch.randn(4, 5, 10)
        >>> v2_mask = torch.ones(4, 5).to(dtype=torch.uint8)
        >>> attention(v1, v2, v2_mask).shape
        torch.Size([4, 5, 20])


    """

    def __init__(self, hidden_size, dropout_rate=0):
        """Init."""
        super().__init__()
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.proj = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, v1, v2, v2_mask):
        """Computing attention vectors and projection vectors."""
        proj_v2 = self.v2_proj(v2)
        similarity_matrix = v1.bmm(proj_v2.transpose(2, 1).contiguous())

        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1).bool(), -1e-7), dim=2)
        v2_wsum = v1_v2_attn.bmm(v2)
        fusion = torch.cat([v1, v2_wsum, v1 - v2_wsum, v1 * v2_wsum], dim=2)
        match = self.dropout(F.relu(self.proj(fusion)))
        return match
