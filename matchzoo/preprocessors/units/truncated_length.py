import typing

import numpy as np

from .unit import Unit


class TruncatedLength(Unit):

    def __init__(
        self,
        text_length: int,
        truncate_mode: str = 'pre'
    ):
        self._text_length = text_length
        self._truncate_mode = truncate_mode

    def transform(self, input_: list) -> list:
        if len(input_) < self._text_length:
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

