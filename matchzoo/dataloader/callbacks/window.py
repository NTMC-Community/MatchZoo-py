from typing import List, Dict, Tuple
from itertools import product, chain, zip_longest

import numpy as np

import matchzoo as mz
from matchzoo.engine.base_callback import BaseCallback


class Window(BaseCallback):
    """
    Generate document match window for each query term.

    :param half_window_size: half of the matching-window size, not including the
        center word, so the full window size is 2 * half_window_size + 1
    :param max_match: a term should have fewer than max_match matching-windows,
        the excess will be discarded

    Example:
        >>> import matchzoo as mz
        >>> from matchzoo.dataloader.callbacks import Ngram
        >>> data = mz.datasets.toy.load_data()
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(ngram_size=3)
        >>> data = preprocessor.fit_transform(data)
        >>> callback = Ngram(preprocessor=preprocessor, mode='index')
        >>> dataset = mz.dataloader.Dataset(
        ...     data, callbacks=[callback])
        >>> _ = dataset[0]

    """

    def __init__(
        self,
        half_window_size: int = 5,
        max_match: int = 20,
    ):
        """Init."""
        self._half_window_size = half_window_size
        self._max_match = max_match

    def on_batch_unpacked(self, x, y):
        """Extract `window_right`, `window_position_right`, `term_window_num` for `x`."""
        batch_size = len(x['text_left'])
        x['window_right'] = [... for _ in range(batch_size)]
        x['window_position_right'] = [... for _ in range(batch_size)]
        x['term_window_num'] = [... for _ in range(batch_size)]
        for idx, (query, query_len, doc, doc_len) in enumerate(zip(
                x['text_left'], x['length_left'], x['text_right'], x['length_right'])):
            window_right, window_position_right, term_window_num = \
                self._build_window(query, query_len, doc, doc_len)
            x['window_right'][idx] = window_right
            x['window_position_right'][idx] = window_position_right
            x['term_window_num'][idx] = term_window_num

        array_query_window_num = np.array([array.shape[0] for array in x['window_right']])
        array_window_right = _pad_sequence(x['window_right'], pad_value=-1)
        array_window_position_right = \
            _pad_sequence(x['window_position_right'], pad_value=-1)
        array_term_window_num = _pad_sequence(x['term_window_num'], pad_value=-1)

        x['query_window_num'] = array_query_window_num
        x['window_right'] = array_window_right
        x['window_position_right'] = array_window_position_right
        x['term_window_num'] = array_term_window_num

    def _build_window(self, query: list, query_len: int, doc: list, doc_len: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        window_of_term = [[] for _ in range(query_len)]
        window_position_of_term = [[] for _ in range(query_len)]
        window_num_of_term = [0 for _ in range(query_len)]

        # get doc window for each query term
        for doc_window_position in range(doc_len):
            padding_doc_window_position = doc_window_position + self._half_window_size
            doc_term = doc[padding_doc_window_position]
            for query_term_position in range(query_len):
                if window_num_of_term[query_term_position] > self._max_match:
                    continue
                query_term = query[query_term_position]
                if query_term == doc_term:
                    window = self._get_window(doc=doc, center=padding_doc_window_position)
                    # window: list, len=full_window_size, element: int, token_id
                    window_of_term[query_term_position].append(window)
                    window_position_of_term[query_term_position].append(
                        doc_window_position)
                    window_num_of_term[query_term_position] += 1

        # window_of_term: list[list[list[int]]]:  len=query_len,
        #   window_of_term[i]: list[list]:  len: window_num of term_i
        #       window_of_term[i][j]: list: len: len=full_window_size,
        #           window_of_term[i][j][k]: int, token_id
        #
        # window_position_of_term: list[list[int]]: len=query_len
        #   window_position_of_term[i]: list[int]: len: window_num
        #       window_position_of_term[i][j]: int, position index of window center
        #
        # window_num_of_term: list[int], len=query_len
        #   window_num_of_term[i]: int, window_num of term_i, sum()

        # flatten
        window_of_term = list(chain.from_iterable(window_of_term))
        window_position_of_term = list(chain.from_iterable(window_position_of_term))

        # to array
        window_of_term = np.stack(window_of_term, axis=0) if len(window_of_term) > 0 \
            else np.zeros((0, 2 * self._half_window_size + 1), dtype=np.long)
        window_position_of_term = np.array(window_position_of_term)
        window_num_of_term = np.array(window_num_of_term)

        return window_of_term, window_position_of_term, window_num_of_term

    def _get_window(self, doc: list, center: int) -> list:
        return doc[center - self._half_window_size: center + self._half_window_size + 1]


def _pad_sequence(list_of_array: List[np.ndarray], pad_value):
    """Padding list of array to an array, like pytorch pad_sequence."""
    batch_size = len(list_of_array)
    max_shape = \
        np.array([array.shape for array in list_of_array]).max(axis=0).tolist()
    batch_array = \
        np.ones([batch_size, *max_shape], dtype=list_of_array[0].dtype) * pad_value
    for i in range(batch_size):
        array = list_of_array[i]
        array_slice = [slice(None, end, None) for end in array.shape]
        batch_array[(i, *array_slice)] = array
    return batch_array
