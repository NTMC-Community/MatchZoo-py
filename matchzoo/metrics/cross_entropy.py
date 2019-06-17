"""CrossEntropy metric for Classification."""
import numpy as np

from matchzoo.engine.base_metric import ClassificationMetric
from matchzoo.utils import one_hot


class CrossEntropy(ClassificationMetric):
    """Cross entropy metric."""

    ALIAS = ['cross_entropy', 'ce']

    def __init__(self):
        """:class:`CrossEntropy` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(
        self,
        y_true: np.array,
        y_pred: np.array,
        eps: float = 1e-12
    ) -> float:
        """
        Calculate cross entropy.

        Example:
            >>> y_true = [0, 1]
            >>> y_pred = [[0.25, 0.25], [0.01, 0.90]]
            >>> CrossEntropy()(y_true, y_pred)
            0.7458274358333028

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :param eps: The Log loss is undefined for p=0 or p=1,
            so probabilities are clipped to max(eps, min(1 - eps, p)).
        :return: Average precision.
        """
        y_pred = np.clip(y_pred, eps, 1. - eps)
        y_true = [
            one_hot(y, num_classes=y_pred.shape[1]) for y in y_true
        ]
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_pred.shape[0]
