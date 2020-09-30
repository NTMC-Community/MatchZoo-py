import collections
import typing

import numpy as np

from .stateful_unit import StatefulUnit


class FrequencyCounter(StatefulUnit):
    """
    Frequency counter unit.

    Examples::
        >>> from collections import Counter
        >>> import matchzoo as mz

    To filter based on document frequency (df):
        >>> unit = mz.preprocessors.units.FrequencyCounter()
        >>> unit.fit([['A', 'B'], ['B', 'C']])
        >>> unit.context
        {'tf': Counter({'B': 2, 'A': 1, 'C': 1}),
         'df': Counter({'B': 2, 'A': 1, 'C': 1}),
         'idf': Counter({'A': 1.4054651081081644, 'C': 1.4054651081081644, 'B': 1.0})}

    """

    def __init__(self):
        """Frequency counter unit."""
        super().__init__()

    def fit(self, list_of_tokens: typing.List[typing.List[str]]):
        """Fit `list_of_tokens` by calculating tf/df/idf states."""

        self._context["tf"] = self._tf(list_of_tokens)
        self._context["df"] = self._df(list_of_tokens)
        self._context["idf"] = self._idf(list_of_tokens)

    def transform(self, input_: list) -> list:
        """Transform do nothing."""
        return input_

    @classmethod
    def _tf(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(tokens)
        return stats

    @classmethod
    def _df(cls, list_of_tokens: list) -> dict:
        stats = collections.Counter()
        for tokens in list_of_tokens:
            stats.update(set(tokens))
        return stats

    @classmethod
    def _idf(cls, list_of_tokens: list) -> dict:
        num_docs = len(list_of_tokens)
        stats = cls._df(list_of_tokens)
        for key, val in stats.most_common():
            stats[key] = np.log((1 + num_docs) / (1 + val)) + 1
        return stats


