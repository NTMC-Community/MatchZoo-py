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
        >>> x_mask = torch.BoolTensor(4, 5)
        >>> attention(x, x_mask).shape
        torch.Size([4, 5])

    """

    def __init__(self, input_size: int = 100):
        """Attention constructor."""
        super().__init__()
        self.linear = nn.Linear(input_size, 1, bias=False)

    def forward(self, x, x_mask):
        """Perform attention on the input."""
        x = self.linear(x).squeeze(dim=-1)
        x = x.masked_fill(x_mask, -float('inf'))
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


class DynamicClipAttention(nn.Module):
    """Computing the dynamic clip attention between two sequence.

    :param clip_type: Type of clip attention. `max` (i.e., Kmax attention)
        indicates keeping top K larggest attention weights, and
        `threshold` (i.e., Kthreshold attention) indicates retaining
        the attention weights that are above K.
    :param topk: When clip_type is `max`, topk is a tuple of two integer
        which represents the left and the right sequence keeping top K
        attention weights; When clip_type is `threshold`, topk should be
        set as None.
    :param threshold: When clip_type is `threshold`, threshold is
        a tuple of two floating-point number which represents
        the left and the right sequence retaining attention weights above
        threshold K; When clip_type is `max`, threshold should be set as None.

    Examples:
        >>> import torch
        >>> attention = DynamicClipAttention(clip_type='max', topk=(10, 5))
        >>> v1 = torch.randn(4, 5, 10)
        >>> v1_mask = torch.ones(4, 5).to(dtype=torch.uint8)
        >>> v2 = torch.randn(4, 10, 10)
        >>> v2_mask = torch.ones(4, 10).to(dtype=torch.uint8)
        >>> attended_v1, attended_v2 = attention(v1, v1_mask, v2, v2_mask)
        >>> attended_v1.shape
        torch.Size([4, 5, 10])

    """

    def __init__(
        self,
        clip_type: str = 'max',
        topk: typing.Union[None, typing.Tuple[int, int]] = (10, 5),
        threshold: typing.Union[None, typing.Tuple[float, float]] = None
    ):
        """Init."""
        super().__init__()
        valid_clip_type = ['max', 'threshold']
        if clip_type not in valid_clip_type:
            raise ValueError(f"{clip_type} is not a valid clip type, "
                             f"{valid_clip_type} expected.")
        if clip_type == "max":
            left_topk, right_topk = topk
            self.left_topk = left_topk
            self.right_topk = right_topk
        else:
            left_threshold, right_threshold = threshold
            self.left_threshold = left_threshold
            self.right_threshold = right_threshold

        self.clip_type = clip_type

    def forward(self, v1, v1_mask, v2, v2_mask):
        """Forward."""
        similarity_matrix = v1.bmm(v2.transpose(2, 1).contiguous())

        v1_v2_attn = F.softmax(
            similarity_matrix.masked_fill(
                v2_mask.unsqueeze(1), -float('inf')), dim=2)
        v2_v1_attn = F.softmax(
            similarity_matrix.masked_fill(
                v1_mask.unsqueeze(2), -float('inf')), dim=1)

        if self.clip_type == "max":
            if self.left_topk > v2.shape[1]:
                raise ValueError(f"left topk should be not larger than the length of "
                                 f"right sequence.")
            topk_attn, topk_attn_index = torch.topk(v1_v2_attn, self.left_topk, dim=2)
            zero_attn = torch.zeros(v1_v2_attn.shape).type_as(v1_v2_attn)
            expand_topk_attn = zero_attn.scatter(2, topk_attn_index, topk_attn)
            attended_v1 = expand_topk_attn.bmm(v2)
        else:
            soft_attn = nn.Threshold(self.left_threshold, 0)(v1_v2_attn)
            attended_v1 = soft_attn.bmm(v2)

        if self.clip_type == "max":
            if self.right_topk > v1.shape[1]:
                raise ValueError(f"right topk should be not larger than the length of "
                                 f"left sequence.")
            v2_v1_attn = v2_v1_attn.transpose(1, 2)
            topk_attn, topk_attn_index = torch.topk(v2_v1_attn, self.right_topk, dim=2)
            zero_attn = torch.zeros(v2_v1_attn.shape).type_as(v2_v1_attn)
            expand_topk_attn = zero_attn.scatter(2, topk_attn_index, topk_attn)
            attended_v2 = expand_topk_attn.bmm(v1)
        else:
            soft_attn = nn.Threshold(self.right_threshold, 0)(v2_v1_attn)
            attended_v2 = soft_attn.transpose(1, 2).bmm(v1)

        attended_v1.masked_fill_(v1_mask.unsqueeze(2), 0)
        attended_v2.masked_fill_(v2_mask.unsqueeze(2), 0)

        return attended_v1, attended_v2
