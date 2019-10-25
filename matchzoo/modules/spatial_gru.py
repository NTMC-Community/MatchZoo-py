"""Spatial GRU module."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from matchzoo.utils import parse_activation


class SpatialGRU(nn.Module):
    """
    Spatial GRU Module.

    :param channels: Number of word interaction tensor channels.
    :param units: Number of SpatialGRU units.
    :param activation: Activation function to use, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: hyperbolic tangent (`tanh`).
    :param recurrent_activation: Activation function to use for
        the recurrent step, one of:
            - String: name of an activation
            - Torch Modele subclass
            - Torch Module instance
            Default: sigmoid activation (`sigmoid`).
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.

    Examples:
        >>> import matchzoo as mz
        >>> channels, units= 4, 10
        >>> spatial_gru = mz.modules.SpatialGRU(channels, units)

    """

    def __init__(
        self,
        channels: int = 4,
        units: int = 10,
        activation: typing.Union[str, typing.Type[nn.Module], nn.Module] = 'tanh',
        recurrent_activation: typing.Union[
            str, typing.Type[nn.Module], nn.Module] = 'sigmoid',
        direction: str = 'lt'
    ):
        """:class:`SpatialGRU` constructor."""
        super().__init__()
        self._units = units
        self._activation = parse_activation(activation)
        self._recurrent_activation = parse_activation(recurrent_activation)
        self._direction = direction
        self._channels = channels

        if self._direction not in ('lt', 'rb'):
            raise ValueError(f"Invalid direction. "
                             f"`{self._direction}` received. "
                             f"Must be in `lt`, `rb`.")

        self._input_dim = self._channels + 3 * self._units

        self._wr = nn.Linear(self._input_dim, self._units * 3)
        self._wz = nn.Linear(self._input_dim, self._units * 4)
        self._w_ij = nn.Linear(self._channels, self._units)
        self._U = nn.Linear(self._units * 3, self._units, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_normal_(self._wr.weight)
        nn.init.xavier_normal_(self._wz.weight)
        nn.init.orthogonal_(self._w_ij.weight)
        nn.init.orthogonal_(self._U.weight)

    def softmax_by_row(self, z: torch.tensor) -> tuple:
        """Conduct softmax on each dimension across the four gates."""

        # z_transform: [B, 4, U]
        z_transform = z.reshape((-1, 4, self._units))
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = F.softmax(z_transform, dim=1).unbind(dim=1)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(
        self,
        inputs: torch.tensor,
        states: list,
        i: int,
        j: int
    ):
        """
        Calculate recurrent unit.

        :param inputs: A tensor which contains interaction
            between left text and right text.
        :param states: An array of tensors which stores the hidden state
            of every step.
        :param i: Recurrent row index.
        :param j: Recurrent column index.

        """

        # Get hidden state h_diag, h_top, h_left
        # h = [B, U]
        h_diag = states[i][j]
        h_top = states[i][j + 1]
        h_left = states[i + 1][j]

        # Get interaction between word i, j: s_ij
        # s = [B, C]
        s_ij = inputs[i][j]

        # Concatenate h_top, h_left, h_diag, s_ij
        # q = [B, 3*U+C]
        q = torch.cat([torch.cat([h_top, h_left], 1), torch.cat([h_diag, s_ij], 1)], 1)

        # Calculate reset gate
        # r = [B, 3*U]
        r = self._recurrent_activation(self._wr(q))

        # Calculate updating gate
        # z: [B, 4*U]
        z = self._wz(q)

        # Perform softmax
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = self.softmax_by_row(z)

        # Get h_ij_
        # h_ij_ = [B, U]
        h_ij_l = self._w_ij(s_ij)
        h_ij_r = self._U(r * (torch.cat([h_left, h_top, h_diag], 1)))
        h_ij_ = self._activation(h_ij_l + h_ij_r)

        # Calculate h_ij
        # h_ij = [B, U]
        h_ij = zl * h_left + zt * h_top + zd * h_diag + zi * h_ij_

        return h_ij

    def forward(self, inputs):
        """
        Perform SpatialGRU on word interation matrix.

        :param inputs: input tensors.
        """

        batch_size, channels, left_length, right_length = inputs.shape

        # inputs = [L, R, B, C]
        inputs = inputs.permute([2, 3, 0, 1])
        if self._direction == 'rb':
            # inputs = [R, L, B, C]
            inputs = torch.flip(inputs, [0, 1])

        # states = [L+1, R+1, B, U]
        states = [
            [torch.zeros([batch_size, self._units]).type_as(inputs)
             for j in range(right_length + 1)] for i in range(left_length + 1)
        ]

        # Calculate h_ij
        # h_ij = [B, U]
        for i in range(left_length):
            for j in range(right_length):
                states[i + 1][j + 1] = self.calculate_recurrent_unit(inputs, states, i, j)
        return states[left_length][right_length]
