"""Semantic composite module for DIIN model."""
import typing

import torch
import torch.nn as nn


class SemanticComposite(nn.Module):
    """
    SemanticComposite module.

    Apply a self-attention layer and a semantic composite fuse gate to compute the
    encoding result of one tensor.

    :param in_features: Feature size of input.
    :param dropout_rate: The dropout rate.

    Examples:
        >>> import torch
        >>> module = SemanticComposite(in_features=10)
        >>> x = torch.randn(4, 5, 10)
        >>> x.shape
        torch.Size([4, 5, 10])
        >>> module(x).shape
        torch.Size([4, 5, 10])

    """

    def __init__(self, in_features, dropout_rate: float = 0.0):
        """Init."""
        super().__init__()
        self.att_linear = nn.Linear(3 * in_features, 1, False)
        self.z_gate = nn.Linear(2 * in_features, in_features, True)
        self.r_gate = nn.Linear(2 * in_features, in_features, True)
        self.f_gate = nn.Linear(2 * in_features, in_features, True)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Forward."""
        seq_length = x.shape[1]

        x_1 = x.unsqueeze(dim=2).repeat(1, 1, seq_length, 1)
        x_2 = x.unsqueeze(dim=1).repeat(1, seq_length, 1, 1)
        x_concat = torch.cat([x_1, x_2, x_1 * x_2], dim=-1)

        # Self-attention layer.
        x_concat = self.dropout(x_concat)
        attn_matrix = self.att_linear(x_concat).squeeze(dim=-1)
        attn_weight = torch.softmax(attn_matrix, dim=2)
        attn = torch.bmm(attn_weight, x)

        # Semantic composite fuse gate.
        x_attn_concat = self.dropout(torch.cat([x, attn], dim=-1))
        x_attn_concat = torch.cat([x, attn], dim=-1)
        z = torch.tanh(self.z_gate(x_attn_concat))
        r = torch.sigmoid(self.r_gate(x_attn_concat))
        f = torch.sigmoid(self.f_gate(x_attn_concat))
        encoding = r * x + f * z

        return encoding
