"""Residual module."""
import math
import typing

import torch
import torch.nn as nn


class AugmentedResidual(nn.Module):
    """Augmented Residual."""

    def forward(self, x, res, i):
        """Forward."""
        if i == 1:
            # res is embedding
            return torch.cat([x, res], dim=-1)

        # latter half of res is embedding
        hidden_size = x.shape[-1]
        x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
        return torch.cat([x, res[:, :, hidden_size:]], dim=-1)
