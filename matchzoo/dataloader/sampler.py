"""Sampler class for dataloader."""
import typing

import math
import numpy as np
from torch.utils.data import Sampler, Dataset

import matchzoo as mz


class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.

    :param dataset: The dataset to sample from.
    """

    def __init__(self, dataset: Dataset):
        """Init."""
        self._dataset = dataset

    def __iter__(self):
        """Get the indices of a batch."""
        return iter(self._dataset.index_pool)

    def __len__(self):
        """Get the total number of instances."""
        return len(self._dataset)


class SortedSampler(Sampler):
    """
    Samples elements according to `length_right`.

    :param dataset: The dataset to sample from.
    """

    def __init__(self, dataset: Dataset):
        """Init."""
        self._dataset = dataset

    def __iter__(self):
        """Get the indices of a batch."""
        self._dataset.sort()
        return iter(self._dataset.index_pool)

    def __len__(self):
        """Get the total number of instances."""
        return len(self._dataset)


class RandomSampler(Sampler):
    """
    Samples elements randomly.

    :param dataset: The dataset to sample from.
    """

    def __init__(self, dataset: Dataset):
        """Init."""
        self._dataset = dataset

    def __iter__(self):
        """Get the indices of a batch."""
        self._dataset.shuffle()
        return iter(self._dataset.index_pool)

    def __len__(self):
        """Get the total number of instances."""
        return len(self._dataset)


class BatchSampler(Sampler):
    """
    Wraps another sampler to yield the indices of a batch.

    :param sampler: Base sampler.
    :param batch_size: Size of a batch.
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int = 32,
    ):
        """Init."""
        self._sampler = sampler
        self._batch_size = batch_size

    def __iter__(self):
        """Get the indices of a batch."""
        batch = []
        for idx in self._sampler:
            batch.append(idx)
            if len(batch) == self._batch_size:
                batch = sum(batch, [])
                yield batch
                batch = []
        if len(batch) > 0:
            batch = sum(batch, [])
            yield batch

    def __len__(self):
        """Get the total number of batch."""
        return math.ceil(len(self._sampler) / self._batch_size)
