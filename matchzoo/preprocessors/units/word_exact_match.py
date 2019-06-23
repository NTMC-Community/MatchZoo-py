import numpy as np

from .unit import Unit


class WordExactMatch(Unit):
    """
    WordExactUnit Class.

    Process unit to get a binary match list of two word index lists. The
    word index list is the word representation of a text.

    Examples:
        >>> import pandas
        >>> input_ = pandas.DataFrame({
        ...  'text_left':[[1, 2, 3],[4, 5, 7, 9]],
        ...  'text_right':[[5, 3, 2, 7],[2, 3, 5]]}
        ... )
        >>> left_word_exact_match = WordExactMatch(
        ...     match='text_left', to_match='text_right'
        ... )
        >>> left_out = input_.apply(left_word_exact_match.transform, axis=1)
        >>> left_out[0]
        [0, 1, 1]
        >>> left_out[1]
        [0, 1, 0, 0]
        >>> right_word_exact_match = WordExactMatch(
        ...     match='text_right', to_match='text_left'
        ... )
        >>> right_out = input_.apply(right_word_exact_match.transform, axis=1)
        >>> right_out[0]
        [0, 1, 1, 0]
        >>> right_out[1]
        [0, 0, 1]

    """

    def __init__(
        self,
        match: str,
        to_match: str
    ):
        """
        Class initialization.

        :param match: the 'match' column name.
        :param to_match: the 'to_match' column name.
        """
        self._match = match
        self._to_match = to_match

    def transform(self, input_) -> list:
        """
        Transform two word index lists into a binary match list.

        :param input_: a dataframe include 'match' column and
            'to_match' column.

        :return: a binary match result list of two word index lists.
        """
        match_binary = []
        for i in range(len(input_[self._match])):
            if input_[self._match][i] in set(input_[self._to_match]):
                match_binary.append(1)
            else:
                match_binary.append(0)

        return match_binary
