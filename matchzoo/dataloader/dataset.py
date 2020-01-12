"""A basic class representing a Dataset."""
import typing
import math
from collections import Iterable

import numpy as np
import pandas as pd
from torch.utils import data

import matchzoo as mz
from matchzoo.engine.base_callback import BaseCallback


class Dataset(data.Dataset):
    """
    Dataset that is built from a data pack.

    :param data_pack: DataPack to build the dataset.
    :param mode: One of "point", "pair", and "list". (default: "point")
    :param num_dup: Number of duplications per instance, only effective when
        `mode` is "pair". (default: 1)
    :param num_neg: Number of negative samples per instance, only effective
        when `mode` is "pair". (default: 1)
    :param batch_size: Batch size. (default: 32)
    :param resample: Either to resample for each epoch, only effective when
        `mode` is "pair". (default: `True`)
    :param shuffle: Either to shuffle the samples/instances. (default: `True`)
    :param sort: Whether to sort data according to length_right. (default: `False`)
    :param callbacks: Callbacks. See `matchzoo.dataloader.callbacks` for more details.

    Examples:
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_data(stage='train')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset_point = mz.dataloader.Dataset(
        ...     data_processed, mode='point', batch_size=32)
        >>> len(dataset_point)
        4
        >>> dataset_pair = mz.dataloader.Dataset(
        ...     data_processed, mode='pair', num_dup=2, num_neg=2, batch_size=32)
        >>> len(dataset_pair)
        1

    """

    def __init__(
        self,
        data_pack: mz.DataPack,
        mode='point',
        num_dup: int = 1,
        num_neg: int = 1,
        batch_size: int = 32,
        resample: bool = False,
        shuffle: bool = True,
        sort: bool = False,
        callbacks: typing.List[BaseCallback] = None
    ):
        """Init."""
        if callbacks is None:
            callbacks = []

        if mode not in ('point', 'pair', 'list'):
            raise ValueError(f"{mode} is not a valid mode type."
                             f"Must be one of `point`, `pair` or `list`.")

        if shuffle and sort:
            raise ValueError(f"parameters `shuffle` and `sort` conflict, "
                             f"should not both be `True`.")

        data_pack = data_pack.copy()
        self._mode = mode
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._batch_size = batch_size
        self._resample = (resample if mode != 'point' else False)
        self._shuffle = shuffle
        self._sort = sort
        self._orig_relation = data_pack.relation
        self._callbacks = callbacks

        if mode == 'pair':
            data_pack.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=num_dup,
                num_neg=num_neg
            )

        self._data_pack = data_pack
        self._batch_indices = None

        self.reset_index()

    def __getitem__(self, item) -> typing.Tuple[dict, np.ndarray]:
        """Get a batch from index idx.

        :param item: the index of the batch.
        """
        if isinstance(item, slice):
            indices = sum(self._batch_indices[item], [])
        elif isinstance(item, Iterable):
            indices = [self._batch_indices[i] for i in item]
        else:
            indices = self._batch_indices[item]
        batch_data_pack = self._data_pack[indices]
        self._handle_callbacks_on_batch_data_pack(batch_data_pack)
        x, y = batch_data_pack.unpack()
        self._handle_callbacks_on_batch_unpacked(x, y)
        return x, y

    def __len__(self) -> int:
        """Get the total number of batches."""
        return len(self._batch_indices)

    def __iter__(self):
        """Create a generator that iterate over the Batches."""
        if self._resample or self._shuffle:
            self.on_epoch_end()
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        """Reorganize the index array if needed."""
        if self._resample:
            self.resample_data()
        self.reset_index()

    def resample_data(self):
        """Reorganize data."""
        if self.mode != 'point':
            self._data_pack.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=self._num_dup,
                num_neg=self._num_neg
            )

    def reset_index(self):
        """
        Set the :attr:`_batch_indices`.

        Here the :attr:`_batch_indices` records the index of all the instances.
        """
        # index pool: index -> instance index
        if self._mode == 'point':
            num_instances = len(self._data_pack)
            index_pool = list(range(num_instances))
        elif self._mode == 'pair':
            index_pool = []
            step_size = self._num_neg + 1
            num_instances = int(len(self._data_pack) / step_size)
            for i in range(num_instances):
                lower = i * step_size
                upper = (i + 1) * step_size
                indices = list(range(lower, upper))
                if indices:
                    index_pool.append(indices)
        elif self._mode == 'list':
            raise NotImplementedError(
                f'{self._mode} dataset not implemented.')
        else:
            raise ValueError(f"{self._mode} is not a valid mode type"
                             f"Must be one of `point`, `pair` or `list`.")

        if self._shuffle:
            np.random.shuffle(index_pool)

        if self._sort:
            old_index_pool = index_pool

            max_instance_right_length = []
            for row in range(len(old_index_pool)):
                instance = self._data_pack[old_index_pool[row]].unpack()[0]
                max_instance_right_length.append(max(instance['length_right']))
            sort_index = np.argsort(max_instance_right_length)

            index_pool = [old_index_pool[index] for index in sort_index]

        # batch_indices: index -> batch of indices
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            if self._mode == 'pair':
                candidates = sum(candidates, [])
            self._batch_indices.append(candidates)

    def _handle_callbacks_on_batch_data_pack(self, batch_data_pack):
        for callback in self._callbacks:
            callback.on_batch_data_pack(batch_data_pack)

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        for callback in self._callbacks:
            callback.on_batch_unpacked(x, y)

    @property
    def callbacks(self):
        """`callbacks` getter."""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value):
        """`callbacks` setter."""
        self._callbacks = value

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value
        self.resample_data()
        self.reset_index()

    @property
    def num_dup(self):
        """`num_dup` getter."""
        return self._num_dup

    @num_dup.setter
    def num_dup(self, value):
        """`num_dup` setter."""
        self._num_dup = value
        self.resample_data()
        self.reset_index()

    @property
    def mode(self):
        """`mode` getter."""
        return self._mode

    @property
    def batch_size(self):
        """`batch_size` getter."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """`batch_size` setter."""
        self._batch_size = value
        self.reset_index()

    @property
    def shuffle(self):
        """`shuffle` getter."""
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        """`shuffle` setter."""
        self._shuffle = value
        self.reset_index()

    @property
    def sort(self):
        """`sort` getter."""
        return self._sort

    @sort.setter
    def sort(self, value):
        """`sort` setter."""
        self._sort = value
        self.reset_index()

    @property
    def resample(self):
        """`resample` getter."""
        return self._resample

    @resample.setter
    def resample(self, value):
        """`resample` setter."""
        self._resample = value
        self.reset_index()

    @property
    def batch_indices(self):
        """`batch_indices` getter."""
        return self._batch_indices

    @classmethod
    def _reorganize_pair_wise(
        cls,
        relation: pd.DataFrame,
        num_dup: int = 1,
        num_neg: int = 1
    ):
        """Re-organize the data pack as pair-wise format."""
        pairs = []
        groups = relation.sort_values(
            'label', ascending=False).groupby('id_left')
        for _, group in groups:
            labels = group.label.unique()
            for label in labels[:-1]:
                pos_samples = group[group.label == label]
                pos_samples = pd.concat([pos_samples] * num_dup)
                neg_samples = group[group.label < label]
                for _, pos_sample in pos_samples.iterrows():
                    pos_sample = pd.DataFrame([pos_sample])
                    neg_sample = neg_samples.sample(num_neg, replace=True)
                    pairs.extend((pos_sample, neg_sample))
        new_relation = pd.concat(pairs, ignore_index=True)
        return new_relation
