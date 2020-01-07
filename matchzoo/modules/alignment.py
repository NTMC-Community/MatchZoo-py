"""Alignment module."""
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class Alignment(nn.Module):
    """Alignment module."""

    def __init__(self, hidden_size: int = 100):
        """Alignment constructor."""
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(1 / math.sqrt(hidden_size)))

    def forward(self, x, y, x_mask, y_mask):
        """Forward."""
        attn = torch.matmul(x, y.transpose(1, 2)) * self.temperature

        mask = torch.matmul(
            x_mask.unsqueeze(dim=2).float(),
            y_mask.unsqueeze(dim=1).float()
        ).bool()

        attn.masked_fill_(mask, -float('inf'))
        attn_x = F.softmax(attn, dim=1)
        attn_y = F.softmax(attn, dim=2)

        feature_y = torch.matmul(attn_x.transpose(1, 2), x)
        feature_x = torch.matmul(attn_y, y)
        return feature_x, feature_y
