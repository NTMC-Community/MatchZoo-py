"""Spatial GRU module."""
import typing

import torch
from torch import nn
import torch.nn.functional as F
from matchzoo.utils import parse_activation


class SpatialGRU(nn.Module):
    """
    Spatial GRU Module.

    :param channels: Number of word interaction tensor channels
    :param units: Number of SpatialGRU units.
    :param activation: Activation function to use. Default:
        hyperbolic tangent (`tanh`). If you pass `None`, no
        activation is applied (ie. "linear" activation: `a(x) = x`).
    :param recurrent_activation: Activation function to use for
        the recurrent step. Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied (ie. "linear"
        activation: `a(x) = x`)
    :param direction: Scanning direction. `lt` (i.e., left top)
        indicates the scanning from left top to right bottom, and
        `rb` (i.e., right bottom) indicates the scanning from
        right bottom to left top.

    Examples:
        >>> import matchzoo as mz
        >>> channels, units= 4, 10
        >>> spatial_gru = mz.modules.SpatialGRU(channels,units)

    """
    def __init__( 
        self,
        channels: int = 4,
        units: int = 10,
        activation: str = 'tanh',
        recurrent_activation: str = 'sigmoid',
        direction: str = 'lt'
    ):
        """:class:`SpatialGRU` constructor."""
        super().__init__()
        self._units = units
        self._activation = parse_activation(activation)
        self._recurrent_activation = parse_activation(recurrent_activation)
        self._direction = direction
        self._channels = channels

        kernel_initializer = 'glorot_uniform'
        recurrent_initializer  = 'orthogonal'

        self._input_dim = self._channels + 3 * self._units
        
        self._wr = self._make_inited_parameter([self._input_dim, self._units * 3], kernel_initializer)
        self._br = self._make_inited_parameter([self._units * 3,], "zeros")
        self._wz = self._make_inited_parameter([self._input_dim, self._units * 4], kernel_initializer)
        self._w_ij = self._make_inited_parameter([self._channels, self._units], recurrent_initializer)
        self._bz = self._make_inited_parameter([self._units * 4,], "zeros")
        self._b_ij = self._make_inited_parameter([self._units,], "zeros")
        self._U = self._make_inited_parameter([self._units * 3, self._units], recurrent_initializer)
    
    @classmethod   
    def _make_inited_parameter(self, shape, initializer):
        """Create and initiate Parameters"""
        W = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        if initializer == "glorot_uniform":
            return nn.init.xavier_uniform_(W)
        elif initializer == "zeros":
            return W
        else:
            return nn.init.orthogonal_(W)

    @classmethod
    def _time_distributed_dense(cls, w, x, b):
        x = torch.matmul(x, w)
        x = x + b 
        return x
    
    def softmax_by_row(self, z: typing.Any) -> tuple:
        """Conduct softmax on each dimension across the four gates."""
        
        # z_transform: [B, 4, U]
        z_transform = z.reshape((-1, 4, self._units))
         # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = F.softmax(z_transform,dim=1).unbind(dim=1)
        return zi, zl, zt, zd

    def calculate_recurrent_unit(
        self,
        inputs: typing.Any,
        states: typing.Any,
        i: int,
        j: int
    ):
        """
        Calculate recurrent unit.

        :param inputs: A TensorArray which contains interaction
            between left text and right text.
        :param states: A TensorArray which stores the hidden state
            of every step.
        :param i: Recurrent row index
        :param j: Recurrent column index

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
        q = torch.cat([torch.cat([h_top, h_left], 1),
        torch.cat([h_diag, s_ij], 1)], 1)

        # Calculate reset gate
        # r = [B, 3*U]
        r = self._recurrent_activation(
            self._time_distributed_dense(self._wr, q, self._br))

        # Calculate updating gate
        # z: [B, 4*U]
        z = self._time_distributed_dense(self._wz, q, self._bz)


        # Perform softmax
        # zi, zl, zt, zd: [B, U]
        zi, zl, zt, zd = self.softmax_by_row(z)

        
        # Get h_ij_
        # h_ij_ = [B, U]
        h_ij_l = self._time_distributed_dense(self._w_ij, s_ij, self._b_ij)
        h_ij_r = torch.matmul(r * (torch.cat([h_left, h_top, h_diag], 1)), self._U)
        h_ij_ = self._activation(h_ij_l + h_ij_r)
        

        # Calculate h_ij
        # h_ij = [B, U]
        h_ij = zl * h_left + zt * h_top + zd * h_diag + zi * h_ij_

        return h_ij

    def forward(self, inputs):
        """
        Perform SpatialGRU on word interation matrix

        :param inputs: input tensors.
        """

        batch_size, channels, left_maxlen, right_maxlen = inputs.shape

        # inputs = [L, R, B, C]
        inputs = inputs.permute([2, 3, 0, 1])
        if self._direction == 'rb':
            # input_x: [R, L, B, C]
            inputs = torch.flip(input_x, [0, 1])
        elif self._direction != 'lt':
            raise ValueError(f"Invalid direction. "
                             f"`{self._direction}` received. "
                             f"Must be in `lt`, `rb`.")

        ## states [L, R, B, U]
        states = [
            [torch.zeros([batch_size, self._units]).type_as(inputs)\
             for j in range(right_maxlen +1)] for i in range(left_maxlen +1)
        ]
        
        ## Calculate h_ij
        # h_ij = [B, U]
        for i in range(left_maxlen):
            for j in range(right_maxlen):
                h_ij = self.calculate_recurrent_unit(inputs, states, i, j)
                states[i + 1][j + 1] = h_ij
        return h_ij
