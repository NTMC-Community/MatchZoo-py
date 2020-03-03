"""Recall for ranking."""
import numpy as np

from matchzoo.engine.base_metric import (
    BaseMetric, sort_and_couple, RankingMetric
)


class Recall(RankingMetric):
    """Recall metric."""

    ALIAS = 'recall'

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`RecallMetric` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}@{self._k}({self._threshold})"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate recall@k.

        Example:
            >>> y_true = [0, 0, 0, 1]
            >>> y_pred = [0.2, 0.4, 0.3, 0.1]
            >>> Recall(k=1)(y_true, y_pred)
            0.0
            >>> Recall(k=2)(y_true, y_pred)
            0.0
            >>> Recall(k=4)(y_true, y_pred)
            1.0
            >>> Recall(k=5)(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Recall @ k.
        :raises: ValueError: k must be greater than 0.
        """
        if self._k <= 0:
            raise ValueError(f"k must be greater than 0."
                             f"{self._k} received.")
        result = 0.
        pos = 0.
        coupled_pair = sort_and_couple(y_true, y_pred)
        for idx, (label, score) in enumerate(coupled_pair):
            if label > self._threshold:
                pos += 1.
                if idx < self._k:
                    result += 1.
        if pos == 0:
            return 0.
        else:
            return result / pos
