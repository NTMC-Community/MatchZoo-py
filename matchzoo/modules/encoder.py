"""Encoder module."""
import typing

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder module."""

    def __init__(self, num_encoder_layers, input_size, hidden_size,
                 kernel_size: int = 3, dropout_rate: float = 0.5):
        """Encoder constructor."""
        super().__init__()
        self.encoders = nn.ModuleList(
            [nn.Sequential(
                nn.ConstantPad1d((0, kernel_size - 1), 0),
                nn.Conv1d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size
                ),
                nn.Dropout(p=dropout_rate)
            ) for i in range(num_encoder_layers)])

    def forward(self, x, x_mask):
        """Forward."""
        x = x.transpose(1, 2)
        x_mask = x_mask.unsqueeze(dim=1)
        for encoder in self.encoders:
            x = x.masked_fill_(x_mask, 0.)
            x = encoder(x)
        return x.transpose(1, 2)
