"""Fusion module."""
import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeLU(nn.Module):
    """GeLU module."""

    def forward(self, x):
        """Forward."""
        return F.gelu(x)


class Fusion(nn.Module):
    """Fusion module."""

    def __init__(self, input_size: int = 100,
                 hidden_size: int = 100, activations: bool = True):
        """Fusion constructor."""
        super().__init__()
        modules = [nn.Linear(input_size, hidden_size)]
        if activations:
            modules.append(GeLU())
        self.fusion = nn.Sequential(*modules)

    def forward(self, x):
        """Forward."""
        return self.fusion(torch.cat(x, dim=-1))


class FullFusion(nn.Module):
    """FullFusion module."""

    def __init__(self, input_size, hidden_size, dropout=0.2):
        """Full fusion constructor."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fusion1 = Fusion(input_size * 2, hidden_size, activations=True)
        self.fusion2 = Fusion(input_size * 2, hidden_size, activations=True)
        self.fusion3 = Fusion(input_size * 2, hidden_size, activations=True)
        self.fusion = Fusion(hidden_size * 3, hidden_size, activations=True)

    def forward(self, x, align):
        """Forward."""
        x1 = self.fusion1([x, align])
        x2 = self.fusion2([x, x - align])
        x3 = self.fusion3([x, x * align])
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.dropout(x)
        return self.fusion([x])
