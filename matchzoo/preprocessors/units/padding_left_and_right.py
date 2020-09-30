from .stateful_unit import StatefulUnit


class PaddingLeftAndRight(StatefulUnit):
    """
    Vocabulary class.

    :param pad_value: The string value for the padding position.
    :param oov_value: The string value for the out-of-vocabulary terms.

    Examples:
        >>> unit = PaddingLeftAndRight(
        ...     left_padding_num=5, right_padding_num=3, pad_value=0)
        >>> unit.transform([3, 1, 4, 1, 5])
        [0, 0, 0, 0, 0, 3, 1, 4, 1, 5, 0, 0, 0]

    """

    def __init__(
            self,
            left_padding_num: int,
            right_padding_num: int,
            pad_value: str or int = 0):
        """Vocabulary unit initializer."""
        super().__init__()
        self._context["left_padding_num"] = left_padding_num
        self._context["right_padding_num"] = right_padding_num
        self._context["pad_value"] = pad_value

    def fit(self, tokens: list):
        """Do nothing."""
        pass

    def transform(self, input_: list) -> list:
        """Padding on left and right."""
        return \
            [self._context["pad_value"]] * self._context["left_padding_num"] \
            + input_ \
            + [self._context["pad_value"]] * self._context["right_padding_num"]
