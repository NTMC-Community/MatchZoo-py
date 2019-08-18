"""The rank hinge loss."""
import torch
from torch import nn
import torch.nn.functional as F


class RankHingeLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ['num_neg', 'margin', 'reduction']

    def __init__(self, num_neg: int = 1, margin: float = 1.,
                 reduction: str = 'mean'):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.num_neg = num_neg
        self.margin = margin
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        y_pos = y_pred[::(self.num_neg + 1), :]
        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = y_pred[(neg_idx + 1)::(self.num_neg + 1), :]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)
        y_neg = torch.mean(y_neg, dim=-1, keepdim=True)
        y_true = torch.ones_like(y_pos)
        return F.margin_ranking_loss(
            y_pos, y_neg, y_true,
            margin=self.margin,
            reduction=self.reduction
        )

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value

    @property
    def margin(self):
        """`margin` getter."""
        return self._margin

    @margin.setter
    def margin(self, value):
        """`margin` setter."""
        self._margin = value
