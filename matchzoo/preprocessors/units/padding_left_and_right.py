from .stateful_unit import StatefulUnit


class PaddingLeftAndRight(StatefulUnit):
    """
    Vocabulary class.

    :param pad_value: The string value for the padding position.
    :param oov_value: The string value for the out-of-vocabulary terms.

    Examples:
        >>> vocab = Vocabulary(pad_value='[PAD]', oov_value='[OOV]')
        >>> vocab.fit(['A', 'B', 'C', 'D', 'E'])
        >>> term_index = vocab.state['term_index']
        >>> term_index  # doctest: +SKIP
        {'[PAD]': 0, '[OOV]': 1, 'D': 2, 'A': 3, 'B': 4, 'C': 5, 'E': 6}
        >>> index_term = vocab.state['index_term']
        >>> index_term  # doctest: +SKIP
        {0: '[PAD]', 1: '[OOV]', 2: 'D', 3: 'A', 4: 'B', 5: 'C', 6: 'E'}

        >>> term_index['out-of-vocabulary-term']
        1
        >>> index_term[0]
        '[PAD]'
        >>> index_term[42]
        Traceback (most recent call last):
            ...
        KeyError: 42
        >>> a_index = term_index['A']
        >>> c_index = term_index['C']
        >>> vocab.transform(['C', 'A', 'C']) == [c_index, a_index, c_index]
        True
        >>> vocab.transform(['C', 'A', '[OOV]']) == [c_index, a_index, 1]
        True
        >>> indices = vocab.transform(list('ABCDDZZZ'))
        >>> ' '.join(vocab.state['index_term'][i] for i in indices)
        'A B C D D [OOV] [OOV] [OOV]'

    """

    def __init__(self, left_padding_num: int, right_padding_num: int, pad_value: str or int = 0):
        """Vocabulary unit initializer."""
        super().__init__()
        self._context["left_padding_num"] = left_padding_num
        self._context["right_padding_num"] = right_padding_num
        self._context["pad_value"] = pad_value

    def fit(self, tokens: list):
        """ Do nothing. """
        pass

    def transform(self, input_: list) -> list:
        """Padding on left and right."""
        return [self._context["pad_value"]] * self._context["left_padding_num"] + \
               input_ + \
               [self._context["pad_value"]] * self._context["right_padding_num"]