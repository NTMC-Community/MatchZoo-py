import numpy as np

import matchzoo as mz
from matchzoo.engine.base_callback import BaseCallback


class Ngram(BaseCallback):
    """
    Generate the character n-gram for data.

    :param preprocessor: The fitted :class:`BasePreprocessor` object, which
         contains the n-gram units information.
    :param mode: It can be one of 'index', 'onehot', 'sum' or 'aggregate'.

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
        preprocessor: mz.preprocessors.BasicPreprocessor,
        mode: str = 'index'
    ):
        """Init."""
        self._mode = mode
        self._word_to_ngram = _build_word_ngram_map(
            preprocessor.context['ngram_process_unit'],
            preprocessor.context['ngram_vocab_unit'],
            preprocessor.context['vocab_unit'].state['index_term'],
            mode
        )

    def on_batch_unpacked(self, x, y):
        """Insert `ngram_left` and `ngram_right` to `x`."""
        batch_size = len(x['text_left'])
        x['ngram_left'] = [[] for i in range(batch_size)]
        x['ngram_right'] = [[] for i in range(batch_size)]
        for idx, row in enumerate(x['text_left']):
            for term in row:
                x['ngram_left'][idx].append(self._word_to_ngram[term])
        for idx, row in enumerate(x['text_right']):
            for term in row:
                x['ngram_right'][idx].append(self._word_to_ngram[term])
        if self._mode == 'aggregate':
            x['ngram_left'] = [list(np.sum(row, axis=0))
                               for row in x['ngram_left']]
            x['ngram_right'] = [list(np.sum(row, axis=0))
                                for row in x['ngram_right']]
            x['text_left'] = x['ngram_left']
            x['text_right'] = x['ngram_right']


def _build_word_ngram_map(
    ngram_process_unit: mz.preprocessors.units.NgramLetter,
    ngram_vocab_unit: mz.preprocessors.units.Vocabulary,
    index_term: dict,
    mode: str = 'index'
) -> dict:
    """
    Generate the word to ngram vector mapping.

    :param ngram_process_unit: The fitted :class:`NgramLetter` object.
    :param ngram_vocab_unit: The fitted :class:`Vocabulary` object.
    :param index_term: The index to term mapping dict.
    :param mode:  It be one of 'index', 'onehot', 'sum' or 'aggregate'.

    :return: the word to ngram vector mapping.
    """
    word_to_ngram = {}
    ngram_size = len(ngram_vocab_unit.state['index_term'])
    for idx, word in index_term.items():
        if idx == 0:
            continue
        elif idx == 1:  # OOV
            word_ngram = [1]
        else:
            ngrams = ngram_process_unit.transform([word])
            word_ngram = ngram_vocab_unit.transform(ngrams)
        num_ngrams = len(word_ngram)
        if mode == 'index':
            word_to_ngram[idx] = word_ngram
        elif mode == 'onehot':
            onehot = np.zeros((num_ngrams, ngram_size))
            onehot[np.arange(num_ngrams), word_ngram] = 1
            word_to_ngram[idx] = onehot
        elif mode == 'sum' or mode == 'aggregate':
            onehot = np.zeros((num_ngrams, ngram_size))
            onehot[np.arange(num_ngrams), word_ngram] = 1
            sum_vector = np.sum(onehot, axis=0)
            word_to_ngram[idx] = sum_vector
        else:
            raise ValueError(f'mode error, it should be one of `index`, '
                             f'`onehot`, `sum` or `aggregate`.'
                             )
    return word_to_ngram
