import typing
from collections import Iterable

import numpy as np

from matchzoo.engine.base_callback import BaseCallback


def infer_dtype(value):
    """Infer the dtype for the features.

    It is required as the input is usually array of objects before padding.
    """
    while isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]

    if not isinstance(value, Iterable):
        return np.array(value).dtype

    if value is not None and len(value) > 0 and np.issubdtype(
            np.array(value).dtype, np.generic):
        dtype = np.array(value[0]).dtype
    else:
        dtype = value.dtype

    # Single Precision
    if dtype == np.double:
        dtype = np.float32

    return dtype


def _padding_2D(input, output, mode: str = 'pre'):
    """
    Pad the input 2D-tensor to the output 2D-tensor.

    :param input: The input 2D-tensor contains the origin values.
    :param output: The output is a shapped 2D-tensor which have filled with pad
     value.
    :param mode: The padding model, which can be 'pre' or 'post'.
    """
    batch_size = min(output.shape[0], len(input))
    pad_length = output.shape[1]
    if mode == 'post':
        for i in range(batch_size):
            end_pos = min(len(input[i]), pad_length)
            if end_pos > 0:
                output[i][:end_pos] = input[i][:end_pos]
    elif mode == 'pre':
        for i in range(batch_size):
            start_pos = min(len(input[i]), pad_length)
            if start_pos > 0:
                output[i][-start_pos:] = input[i][-start_pos:]
    else:
        raise ValueError('{} is not a vaild pad mode.'.format(mode))


def _padding_3D(input, output, mode: str = 'pre'):
    """
    Pad the input 3D-tensor to the output 3D-tensor.

    :param input: The input 3D-tensor contains the origin values.
    :param output: The output is a shapped 3D-tensor which have filled with pad
     value.
    :param mode: The padding model, which can be 'pre' or 'post'.
    """
    batch_size = min(output.shape[0], len(input))
    pad_1d_length = output.shape[1]
    pad_2d_length = output.shape[2]
    if mode == 'post':
        for i in range(batch_size):
            len_d1 = min(len(input[i]), pad_1d_length)
            for j in range(len_d1):
                end_pos = min(len(input[i][j]), pad_2d_length)
                if end_pos > 0:
                    output[i][j][:end_pos] = input[i][j][:end_pos]
    elif mode == 'pre':
        for i in range(batch_size):
            len_d1 = min(len(input[i]), pad_1d_length)
            for j in range(len_d1):
                start_pos = min(len(input[i][j]), pad_2d_length)
                if start_pos > 0:
                    output[i][j][-start_pos:] = input[i][j][-start_pos:]
    else:
        raise ValueError('{} is not a vaild pad mode.'.format(mode))


class BasicPadding(BaseCallback):
    """
    Pad data for basic preprocessor.

    :param fixed_length_left: Integer. If set, `text_left` will be padded
        to this length.
    :param fixed_length_right: Integer. If set, `text_right` will be padded
        to this length.
    :param pad_word_value: the value to fill text.
    :param pad_word_mode: String, `pre` or `post`:
        pad either before or after each sequence.
    :param with_ngram: Boolean. Whether to pad the n-grams.
    :param fixed_ngram_length: Integer. If set, each word will be padded to
        this length, or it will be set as the maximum length of words in
        current batch.
    :param pad_ngram_value: the value to fill empty n-grams.
    :param pad_ngram_mode: String, `pre` or `post`: pad either before of after
        each sequence.
    """

    def __init__(
        self,
        fixed_length_left: int = None,
        fixed_length_right: int = None,
        pad_word_value: typing.Union[int, str] = 0,
        pad_word_mode: str = 'pre',
        with_ngram: bool = False,
        fixed_ngram_length: int = None,
        pad_ngram_value: typing.Union[int, str] = 0,
        pad_ngram_mode: str = 'pre'
    ):
        """Init."""
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._pad_word_value = pad_word_value
        self._pad_word_mode = pad_word_mode
        self._with_ngram = with_ngram
        self._fixed_ngram_length = fixed_ngram_length
        self._pad_ngram_value = pad_ngram_value
        self._pad_ngram_mode = pad_ngram_mode

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""

        batch_size = len(x['id_left'])
        pad_length_left = max(x['length_left'])
        pad_length_right = max(x['length_right'])
        if self._with_ngram:
            ngram_length_left = max([len(w)
                                     for k in x['ngram_left'] for w in k])
            ngram_length_right = max([len(w)
                                      for k in x['ngram_right'] for w in k])
            ngram_length = max(ngram_length_left, ngram_length_right)
            if self._fixed_ngram_length:
                ngram_length = self._fixed_ngram_length

        if self._fixed_length_left is not None:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is not None:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            dtype = infer_dtype(value)

            if key == 'text_left':
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_word_value, dtype=dtype)
                _padding_2D(value, padded_value, self._pad_word_mode)
            elif key == 'text_right':
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_word_value, dtype=dtype)
                _padding_2D(value, padded_value, self._pad_word_mode)
            elif key == 'ngram_left':
                padded_value = np.full(
                    [batch_size, pad_length_left, ngram_length],
                    self._pad_ngram_value, dtype=dtype
                )
                _padding_3D(value, padded_value, self._pad_ngram_mode)
            elif key == 'ngram_right':
                padded_value = np.full(
                    [batch_size, pad_length_right, ngram_length],
                    self._pad_ngram_value, dtype=dtype
                )
                _padding_3D(value, padded_value, self._pad_ngram_mode)
            else:
                continue
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
        batch_size = len(x['id_left'])
        pad_length_left = max(x['length_left'])
        pad_length_right = max(x['length_right'])
        bin_size = len(x['match_histogram'][0][0])

        if self._fixed_length_left is not None:
            pad_length_left = self._fixed_length_left
        if self._fixed_length_right is not None:
            pad_length_right = self._fixed_length_right

        for key, value in x.items():
            if key != 'text_left' and key != 'text_right' and \
                    key != 'match_histogram':
                continue

            dtype = infer_dtype(value)

            if key == 'text_left':
                padded_value = np.full([batch_size, pad_length_left],
                                       self._pad_value, dtype=dtype)
                _padding_2D(value, padded_value, self._pad_mode)
            elif key == 'text_right':
                padded_value = np.full([batch_size, pad_length_right],
                                       self._pad_value, dtype=dtype)
                _padding_2D(value, padded_value, self._pad_mode)
            else:  # key == 'match_histogram'
                padded_value = np.full(
                    [batch_size, pad_length_left, bin_size],
                    self._pad_value, dtype=dtype)
                _padding_3D(value, padded_value, self._pad_mode)
            x[key] = padded_value


class BertPadding(BaseCallback):
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
                                     pad_word_value=pad_value,
                                     pad_word_mode=pad_mode)

    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """Pad `x['text_left']` and `x['text_right]`."""
        self._padding.on_batch_unpacked(x, y)
        x['text_left'] = np.insert(x['text_left'], 0, 101, axis=1)
        x['text_left'] = np.insert(x['text_left'], x['text_left'][0].size, 102, axis=1)
        x['text_right'] = np.insert(x['text_right'], x['text_right'][0].size, 102, axis=1)
