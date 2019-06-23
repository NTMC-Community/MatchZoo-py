"""Accuracy metric for Classification."""
import numpy as np

from matchzoo.engine.base_metric import ClassificationMetric


class Accuracy(ClassificationMetric):
    """Accuracy metric."""

    ALIAS = ['accuracy', 'acc']

    def __init__(self):
        """:class:`Accuracy` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate accuracy.

        Example:
            >>> import numpy as np
            >>> y_true = np.array([1])
            >>> y_pred = np.array([[0, 1]])
            >>> Accuracy()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Accuracy.
        """
        y_pred = np.argmax(y_pred, axis=1)
        return np.sum(y_pred == y_true) / float(y_true.size)
