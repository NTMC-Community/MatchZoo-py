"""DenseNet module."""
import typing

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """Dense block of DenseNet."""

    def __init__(
        self,
        in_channels,
        growth_rate: int = 20,
        kernel_size: tuple = (2, 2),
        layers_per_dense_block: int = 3
    ):
        """Init."""
        super().__init__()
        dense_block = []
        for _ in range(layers_per_dense_block):
            conv_block = self._make_conv_block(in_channels, growth_rate, kernel_size)
            dense_block.append(conv_block)
            in_channels += growth_rate
        self._dense_block = nn.ModuleList(dense_block)

    def forward(self, x):
        """Forward."""
        for layer in self._dense_block:
            conv_out = layer(x)
            x = torch.cat([x, conv_out], dim=1)
        return x

    @classmethod
    def _make_conv_block(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple
    ) -> nn.Module:
        """Make conv block."""
        return nn.Sequential(
            nn.ConstantPad2d(
                (0, kernel_size[1] - 1, 0, kernel_size[0] - 1), 0
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size
            ),
            nn.ReLU()
        )


class DenseNet(nn.Module):
    """
    DenseNet module.

    :param in_channels: Feature size of input.
    :param nb_dense_blocks: The number of blocks in densenet.
    :param layers_per_dense_block: The number of convolution layers in dense block.
    :param growth_rate: The filter size of each convolution layer in dense block.
    :param transition_scale_down_ratio: The channel scale down ratio of the convolution
        layer in transition block.
    :param conv_kernel_size: The kernel size of convolution layer in dense block.
    :param pool_kernel_size: The kernel size of pooling layer in transition block.
    """

    def __init__(
        self,
        in_channels,
        nb_dense_blocks: int = 3,
        layers_per_dense_block: int = 3,
        growth_rate: int = 10,
        transition_scale_down_ratio: float = 0.5,
        conv_kernel_size: tuple = (2, 2),
        pool_kernel_size: tuple = (2, 2),
    ):
        """Init."""
        super().__init__()
        dense_blocks = []
        transition_blocks = []
        for _ in range(nb_dense_blocks):
            dense_block = DenseBlock(
                in_channels, growth_rate, conv_kernel_size, layers_per_dense_block)
            in_channels += layers_per_dense_block * growth_rate
            dense_blocks.append(dense_block)

            transition_block = self._make_transition_block(
                in_channels, transition_scale_down_ratio, pool_kernel_size)
            in_channels = int(in_channels * transition_scale_down_ratio)
            transition_blocks.append(transition_block)

        self._dense_blocks = nn.ModuleList(dense_blocks)
        self._transition_blocks = nn.ModuleList(transition_blocks)

        self._out_channels = in_channels

    @property
    def out_channels(self) -> int:
        """`out_channels` getter."""
        return self._out_channels

    def forward(self, x):
        """Forward."""
        for dense_block, trans_block in zip(self._dense_blocks, self._transition_blocks):
            x = dense_block(x)
            x = trans_block(x)
        return x

    @classmethod
    def _make_transition_block(
        cls,
        in_channels: int,
        transition_scale_down_ratio: float,
        pool_kernel_size: tuple
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(in_channels * transition_scale_down_ratio),
                kernel_size=1
            ),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
        )
