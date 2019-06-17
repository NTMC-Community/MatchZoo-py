"""The rank cross entropy loss."""
import torch
from torch import nn
import torch.nn.functional as F
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class RankCrossEntropyLoss(nn.Module):
    """Creates a criterion that measures rank cross entropy loss."""

    __constants__ = ['num_neg']

    def __init__(self, num_neg: int = 1):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        """
        super().__init__()
        self.num_neg = num_neg

    @weak_script_method
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param y_pred: Label.
        :param y_true: Predicted result.
        :return: Hinge loss computed by user-defined margin.
        """
        y_true = torch.unsqueeze(y_true, 1).float()
        logits = y_pred[::(self.num_neg + 1), :]
        labels = y_true[::(self.num_neg + 1), :]
        for neg_idx in range(self.num_neg):
            neg_logits = y_pred[(neg_idx + 1)::(self.num_neg + 1), :]
            neg_labels = y_true[(neg_idx + 1)::(self.num_neg + 1), :]
            logits = torch.cat((logits, neg_logits), dim=-1)
            labels = torch.cat((labels, neg_labels), dim=-1)
        return -torch.mean(
            torch.sum(labels * torch.log(F.softmax(logits, dim=-1)), dim=-1)
        )

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value
