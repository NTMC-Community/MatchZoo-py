import typing

import numpy as np

from .unit import Unit


class TruncatedLength(Unit):
    """
    TruncatedLengthUnit Class.

    Process unit to truncate the text that exceeds the set length.

    Examples:
        >>> from matchzoo.preprocessors.units import TruncatedLength
        >>> truncatedlen = TruncatedLength(3)
        >>> truncatedlen.transform(list(range(1, 6))) == [3, 4, 5]
        True
        >>> truncatedlen.transform(list(range(2))) == [0, 1]
        True

    """

    def __init__(
        self,
        text_length: int,
        truncate_mode: str = 'pre'
    ):
        """
        Class initialization.

        :param text_length: the specified maximum length of text.
        :param truncate_mode: String, `pre` or `post`:
            remove values from sequences larger than :attr:`text_length`,
            either at the beginning or at the end of the sequences.
        """
        self._text_length = text_length
        self._truncate_mode = truncate_mode

    def transform(self, input_: list) -> list:
        """
        Truncate the text that exceeds the specified maximum length.

        :param input_: list of tokenized tokens.

        :return tokens: list of tokenized tokens in fixed length
            if its origin length larger than :attr:`text_length`.
        """
        if len(input_) <= self._text_length:
            truncated_tokens = input_
        else:
            if self._truncate_mode == 'pre':
                truncated_tokens = input_[-self._text_length:]
            elif self._truncate_mode == 'post':
                truncated_tokens = input_[:self._text_length]
            else:
                raise ValueError('{} is not a vaild '
                                 'truncate mode.'.format(self._truncate_mode))
        return truncated_tokens
