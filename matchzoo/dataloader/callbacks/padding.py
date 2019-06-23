import typing

import numpy as np

from matchzoo.dataloader.callbacks import Callback


class BasicPadding(Callback):
    """
    Padding data for basic preprocessor.

    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        batch_size = len(x['id_left'])

        max_left_len = max(x['length_left'])
        max_right_len = max(x['length_right'])

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right':
                continue

            if key == 'text_left':
                padded_value = np.full([batch_size, max_left_len],
                                       self._pad_value, dtype=value.dtype)
            else:  # key == 'text_right'
                padded_value = np.full([batch_size, max_right_len],
                                       self._pad_value, dtype=value.dtype)
            if self._pad_mode == 'post':
                for i in range(len(value)):
                    if len(value[i]) > 0:
                        padded_value[i][:len(value[i])] = value[i]
            elif self._pad_mode == 'pre':
                for i in range(len(value)):
                    if len(value[i]) > 0:
                        padded_value[i][-len(value[i]):] = value[i]
            else:
                raise ValueError('{} is not a vaild '
                                 'pad mode.'.format(self._pad_mode))
            x[key] = padded_value


class DRMMPadding(Callback):
    """
    Pad data for DRMM Model.

    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        Padding.

        Pad `x['text_left']`, `x['text_right]` and `x['match_histogram']`.
        """
        batch_size = len(x['id_left'])

        max_left_len = max(x['length_left'])
        max_right_len = max(x['length_right'])
        bin_size = len(x['match_histogram'][0][0])

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right' and \
                    key != 'match_histogram':
                continue

            if key == 'text_left':
                padded_value = np.full([batch_size, max_left_len],
                                       self._pad_value, dtype=value.dtype)
            elif key == 'text_right':
                padded_value = np.full([batch_size, max_right_len],
                                       self._pad_value, dtype=value.dtype)
            else:  # key == 'match_histogram'
                padded_value = np.full([batch_size, max_left_len, bin_size],
                                       self._pad_value, dtype=value.dtype)

            if self._pad_mode == 'post':
                for i in range(len(value)):
                    if len(value[i]) > 0:
                        padded_value[i][:len(value[i])] = value[i]
            elif self._pad_mode == 'pre':
                for i in range(len(value)):
                    if len(value[i]) > 0:
                        padded_value[i][-len(value[i]):] = value[i]
            else:
                raise ValueError('{} is not a vaild '
                                 'pad mode.'.format(self._pad_mode))
            x[key] = padded_value


class CDSSMPadding(Callback):
    """
    Pad data for cdssm preprocessor.

    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        batch_size = len(x['id_left'])

        max_left_len = max(x['length_left'])
        max_right_len = max(x['length_right'])
        vocab_size = len(x['text_left'][0][1])

        for key, value in x.items():
            if key == 'text_left':
                padded_value = np.full([batch_size, max_left_len, vocab_size],
                                       fill_value=0, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        left_len = np.array(value[i]).shape[0]
                        if left_len > 0:
                            padded_value[i][:left_len][:] = value[i]
                        if left_len < max_left_len:
                            padded_value[i, left_len:, self._pad_value] = \
                                [1] * (max_left_len - left_len)
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        left_len = np.array(value[i]).shape[0]
                        if left_len > 0:
                            padded_value[i][-left_len:][:] = value[i]
                        if left_len < max_left_len:
                            padded_value[i, :-left_len, self._pad_value] = \
                                [1] * (max_left_len - left_len)
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'text_right':
                padded_value = np.full([batch_size, max_right_len, vocab_size],
                                       fill_value=0, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        right_len = np.array(value[i]).shape[0]
                        if right_len > 0:
                            padded_value[i][:right_len][:] = value[i]
                        if right_len < max_right_len:
                            padded_value[i, right_len:, self._pad_value] = \
                                [1] * (max_right_len - right_len)
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        right_len = np.array(value[i]).shape[0]
                        if right_len > 0:
                            padded_value[i][-right_len:][:] = value[i]
                        if right_len < max_right_len:
                            padded_value[i, :-right_len, self._pad_value] = \
                                [1] * (max_right_len - right_len)
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:
                continue
            x[key] = padded_value


class DIINPadding(Callback):
    """
    Pad data for diin preprocessor.

    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        Padding.

        Pad `x['text_left']`, `x['text_right]`,
            `x['char_left']`, `x['char_right]`,
            `x['match_left']`, `x['match_right]`.
        """
        batch_size = len(x['id_left'])

        max_left_len = max(x['length_left'])
        max_right_len = max(x['length_right'])

        max_left_word_len = max(
            [len(i) for i in sum(x['char_left'].tolist(), [])])
        max_right_word_len = max(
            [len(i) for i in sum(x['char_right'].tolist(), [])])

        for key, value in x.items():
            if key == 'text_left' or key == 'match_left' or \
                    key == 'text_right' or key == 'match_right':
                if key == 'text_left' or key == 'match_left':
                    padded_value = np.full([batch_size, max_left_len],
                                           self._pad_value, dtype=value.dtype)
                else:
                    padded_value = np.full([batch_size, max_right_len],
                                           self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        if len(value[i]) > 0:
                            padded_value[i][:len(value[i])] = value[i]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        if len(value) > 0:
                            padded_value[i][-len(value[i]):] = value[i]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'char_left' or key == 'char_right':
                if key == 'char_left':
                    padded_value = np.full(
                        [batch_size, max_left_len, max_left_word_len],
                        self._pad_value, dtype=value.dtype)
                else:
                    padded_value = np.full(
                        [batch_size, max_right_len, max_right_word_len],
                        self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        text_len = len(value[i])
                        for j in range(text_len):
                            word_len = len(value[i][j])
                            if word_len > 0:
                                padded_value[i][j][:word_len] = value[i][j]
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        text_len = len(value[i])
                        for j in range(text_len):
                            word_len = len(value[i][j])
                            if word_len > 0:
                                padded_value[i][-text_len + j][-word_len:] = \
                                    value[i][j]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:
                continue
            x[key] = padded_value
