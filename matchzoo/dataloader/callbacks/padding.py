import typing

import numpy as np

from matchzoo.engine.base_callback import BaseCallback


class BasicPadding(BaseCallback):
    """
    Pad data for basic preprocessor.

    :param fixed_length_left: Integer. If set, `text_left` will be padded
        to this length.
    :param fixed_length_right: Integer. If set, `text_right` will be padded
        to this length.
    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        pad_length_left = 0
        pad_length_right = 0

        batch_size = len(x['id_left'])
        max_length_left = max(x['length_left'])
        max_length_right = max(x['length_right'])

        if self._fixed_length_left is None:
            pad_length_left = max_length_left
        else:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is None:
            pad_length_right = max_length_right
        else:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right':
                continue
            elif key == 'text_left':
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_left)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_left)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:  # key == 'text_right'
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_right)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_right)
                        if len(value[i]) > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            x[key] = padded_value


class DRMMPadding(BaseCallback):
    """
    Pad data for DRMM Model.

    :param fixed_length_left: Integer. If set, `text_left` and
        `match_histogram` will be padded to this length.
    :param fixed_length_right: Integer. If set, `text_right` will be padded
        to this length.
    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        Padding.

        Pad `x['text_left']`, `x['text_right]` and `x['match_histogram']`.
        """
        pad_length_left = 0
        pad_length_right = 0

        batch_size = len(x['id_left'])
        max_length_left = max(x['length_left'])
        max_length_right = max(x['length_right'])
        bin_size = len(x['match_histogram'][0][0])

        if self._fixed_length_left is None:
            pad_length_left = max_length_left
        else:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is None:
            pad_length_right = max_length_right
        else:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right' and \
                    key != 'match_histogram':
                continue
            elif key == 'text_left':
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_left)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_left)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'text_right':
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_right)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_right)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:  # key == 'match_histogram'
                padded_value = np.full(
                    [batch_size, pad_length_left, bin_size],
                    self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_left)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_left)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            x[key] = padded_value


class CDSSMPadding(BaseCallback):
    """
    Pad data for cdssm preprocessor.

    :param fixed_length_left: Integer. If set, `text_left` will be padded
        to this length.
    :param fixed_length_right: Integer. If set, `text_right` will be padded
        to this length.
    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        pad_length_left = 0
        pad_length_right = 0

        batch_size = len(x['id_left'])
        max_length_left = max(x['length_left'])
        max_length_right = max(x['length_right'])
        vocab_size = len(x['text_left'][0][0])

        if self._fixed_length_left is None:
            pad_length_left = max_length_left
        else:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is None:
            pad_length_right = max_length_right
        else:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            if key == 'text_left':
                padded_value = np.full(
                    [batch_size, pad_length_left, vocab_size],
                    fill_value=0, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        left_len = np.array(value[i]).shape[0]
                        end_pos = min(left_len, pad_length_left)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                        if end_pos < pad_length_left:
                            padded_value[i, end_pos:, self._pad_value] = \
                                [1] * (pad_length_left - end_pos)
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        left_len = np.array(value[i]).shape[0]
                        start_pos = min(left_len, pad_length_left)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                        if start_pos < pad_length_left:
                            padded_value[i, :-start_pos, self._pad_value] = \
                                [1] * (pad_length_left - start_pos)
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'text_right':
                padded_value = np.full(
                    [batch_size, pad_length_right, vocab_size],
                    fill_value=0, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        right_len = np.array(value[i]).shape[0]
                        end_pos = min(right_len, pad_length_right)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                        if end_pos < pad_length_right:
                            padded_value[i, end_pos:, self._pad_value] = \
                                [1] * (pad_length_right - end_pos)
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        right_len = np.array(value[i]).shape[0]
                        start_pos = min(right_len, pad_length_right)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                        if start_pos < pad_length_right:
                            padded_value[i, :-start_pos, self._pad_value] = \
                                [1] * (pad_length_right - start_pos)
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:
                continue
            x[key] = padded_value


class DIINPadding(BaseCallback):
    """
    Pad data for diin preprocessor.

    :param fixed_length_left: Integer. If set, `text_left`, `char_left`
        and `match_left` will be padded to this length.
    :param fixed_length_right: Integer. If set, `text_right`, `char_right`
        and `match_right` will be padded to this length.
    :param fixed_length_word: Integer. If set, words in `char_left` and
        `char_right` will be padded to this length.
    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        fixed_length_word: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._fixed_length_word = fixed_length_word
        self._pad_value = pad_value
        self._pad_mode = pad_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        Padding.

        Pad `x['text_left']`, `x['text_right]`,
            `x['char_left']`, `x['char_right]`,
            `x['match_left']`, `x['match_right]`.
        """
        pad_length_left = 0
        pad_length_right = 0
        pad_length_word_left = 0
        pad_length_word_right = 0

        batch_size = len(x['id_left'])
        max_length_left = max(x['length_left'])
        max_length_right = max(x['length_right'])
        max_length_word_left = max(
            [len(i) for i in sum(x['char_left'].tolist(), [])])
        max_length_word_right = max(
            [len(i) for i in sum(x['char_right'].tolist(), [])])

        if self._fixed_length_left is None:
            pad_length_left = max_length_left
        else:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is None:
            pad_length_right = max_length_right
        else:
            pad_length_right = self._fixed_length_right
        if self._fixed_length_word is None:
            pad_length_word_left = max_length_word_left
            pad_length_word_right = max_length_word_right
        else:
            pad_length_word_left = self._fixed_length_word
            pad_length_word_right = self._fixed_length_word

        for key, value in x.items():
            if key == 'text_left' or key == 'match_left':
                if key == 'text_left':
                    padded_value = np.full([batch_size, pad_length_left],
                                           self._pad_value, dtype=value.dtype)
                else:  # key == 'match_left'
                    padded_value = np.full([batch_size, pad_length_left],
                                           fill_value=0, dtype=value.dtype)

                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_left)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_left)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'text_right' or key == 'match_right':
                if key == 'test_right':
                    padded_value = np.full([batch_size, pad_length_right],
                                           self._pad_value, dtype=value.dtype)
                else:  # key == 'match_right'
                    padded_value = np.full([batch_size, pad_length_right],
                                           fill_value=0, dtype=value.dtype)

                if self._pad_mode == 'post':
                    for i in range(len(value)):
                        end_pos = min(len(value[i]), pad_length_right)
                        if end_pos > 0:
                            padded_value[i][:end_pos] = value[i][:end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(len(value)):
                        start_pos = min(len(value[i]), pad_length_right)
                        if start_pos > 0:
                            padded_value[i][-start_pos:] = \
                                value[i][-start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'char_left':
                padded_value = np.full(
                    [batch_size, pad_length_left, pad_length_word_left],
                    self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        text_end_pos = min(len(value[i]), pad_length_left)
                        for j in range(text_end_pos):
                            word_end_pos = min(len(value[i][j]),
                                               pad_length_word_left)
                            if word_end_pos > 0:
                                padded_value[i][j][:word_end_pos] = \
                                    value[i][j][:word_end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        text_start_pos = min(len(value[i]), pad_length_left)
                        for j in range(text_start_pos):
                            word_start_pos = min(len(value[i][j]),
                                                 pad_length_word_left)
                            if word_start_pos > 0:
                                row = -text_start_pos + j
                                padded_value[i][row][-word_start_pos:] = \
                                    value[i][j][-word_start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            elif key == 'char_right':
                padded_value = np.full(
                    [batch_size, pad_length_right, pad_length_word_right],
                    self._pad_value, dtype=value.dtype)
                if self._pad_mode == 'post':
                    for i in range(batch_size):
                        text_end_pos = min(len(value[i]), pad_length_right)
                        for j in range(text_end_pos):
                            word_end_pos = min(len(value[i][j]),
                                               pad_length_word_right)
                            if word_end_pos > 0:
                                padded_value[i][j][:word_end_pos] = \
                                    value[i][j][:word_end_pos]
                elif self._pad_mode == 'pre':
                    for i in range(batch_size):
                        text_start_pos = min(len(value[i]), pad_length_right)
                        for j in range(text_start_pos):
                            word_start_pos = min(len(value[i][j]),
                                                 pad_length_word_right)
                            if word_start_pos > 0:
                                row = -text_start_pos + j
                                padded_value[i][row][-word_start_pos:] = \
                                    value[i][j][-word_start_pos:]
                else:
                    raise ValueError('{} is not a vaild '
                                     'pad mode.'.format(self._pad_mode))
            else:
                continue
            x[key] = padded_value


class BERTPadding(BaseCallback):
    """
    Pad data for bert preprocessor.

    :param fixed_length_left: Integer. If set, `text_left` will be padded
        to this length.
    :param fixed_length_right: Integer. If set, `text_right` will be padded
        to this length.
    :param pad_value: the value to fill text.
    :param pad_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_value: typing.Union[int, str] = 0,
        pad_mode: str = 'pre',
    ):
        """Init."""
        self._padding = BasicPadding(fixed_length_left=fixed_length_left,
                                     fixed_length_right=fixed_length_right,
                                     pad_value=pad_value,
                                     pad_mode=pad_mode)

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        self._padding.on_batch_unpacked(x, y)
        x['text_left'] = np.insert(x['text_left'], 0, 101, axis=1)
        x['text_right'] = np.insert(x['text_right'], 0, 102, axis=1)
        SEP = [[102]] * len(x['text_right'])
        x['text_right'] = np.append(x['text_right'], SEP, axis=1)
