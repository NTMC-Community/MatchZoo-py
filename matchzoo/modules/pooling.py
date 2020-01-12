"""Pooling module."""
import typing

import torch
import torch.nn as nn


class Pooling(nn.Module):
    """Pooling module."""

    def forward(self, x, mask):
        """Forward."""
        return x.masked_fill_(mask.unsqueeze(dim=2), -float('inf')).max(dim=1)[0]
