"""Early stopping."""

import typing

import torch
import numpy as np


class EarlyStopping:
    """
    EarlyStopping stops training if no improvement after a given patience.

    :param patience: Number fo events to wait if no improvement and then
        stop the training.
    :param should_decrease: The way to judge the best so far.
    :param key: Key of metric to be compared.
    """

    def __init__(
        self,
        patience: typing.Optional[int] = None,
        should_decrease: bool = None,
        key: typing.Any = None
    ):
        """Early stopping Constructor."""
        self._patience = patience
        self._key = key
        self._best_so_far = 0
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = False
        self._early_stop = False

    def state_dict(self) -> typing.Dict[str, typing.Any]:
        """A `Trainer` can use this to serialize the state."""
        return {
            'patience': self._patience,
            'best_so_far': self._best_so_far,
            'is_best_so_far': self._is_best_so_far,
            'epochs_with_no_improvement': self._epochs_with_no_improvement,
        }

    def load_state_dict(
        self,
        state_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """Hydrate a early stopping from a serialized state."""
        self._patience = state_dict["patience"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._best_so_far = state_dict["best_so_far"]
        self._epochs_with_no_improvement = \
            state_dict["epochs_with_no_improvement"]

    def update(self, result: list):
        """Call function."""
        score = result[self._key]
        if score > self._best_so_far:
            self._best_so_far = score
            self._is_best_so_far = True
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1

    @property
    def best_so_far(self) -> bool:
        """Returns best so far."""
        return self._best_so_far

    @property
    def is_best_so_far(self) -> bool:
        """Returns true if it is the best so far."""
        return self._is_best_so_far

    @property
    def should_stop_early(self) -> bool:
        """Returns true if improvement has stopped for long enough."""
        if not self._patience:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience
