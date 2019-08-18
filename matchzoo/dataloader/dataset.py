"""A basic class representing a Dataset."""
import typing

import functools
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
    :param callbacks: Callbacks. See `matchzoo.data_generator.callbacks` for
        more details.

    Examples:
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_data(stage='train')
        >>> preprocessor = mz.preprocessors.CDSSMPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset_point = mz.dataloader.Dataset(data_processed, mode='point')
        >>> len(dataset_point)
        100
        >>> dataset_pair = mz.dataloader.Dataset(
        ...     data_processed, mode='pair', num_neg=2)
        >>> len(dataset_pair)
        5

    """

    def __init__(
        self,
        data_pack: mz.DataPack,
        mode='point',
        num_dup: int = 1,
        num_neg: int = 1,
        callbacks: typing.List[BaseCallback] = None
    ):
        """Init."""
        if callbacks is None:
            callbacks = []

        if mode not in ('point', 'pair', 'list'):
            raise ValueError(f"{mode} is not a valid mode type."
                             f"Must be one of `point`, `pair` or `list`.")

        self._mode = mode
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._orig_relation = data_pack.relation
        self._callbacks = callbacks
        self._data_pack = data_pack
        self._index_pool = None
        self.sample()

    def __len__(self) -> int:
        """Get the total number of instances."""
        return len(self._index_pool)

    def __getitem__(self, item: int) -> typing.Tuple[dict, np.ndarray]:
        """Get a set of instances from index idx.

        :param item: the index of the instance.
        """
        item_data_pack = self._data_pack[item]
        self._handle_callbacks_on_batch_data_pack(item_data_pack)
        x, y = item_data_pack.unpack()
        self._handle_callbacks_on_batch_unpacked(x, y)
        return x, y

    def _handle_callbacks_on_batch_data_pack(self, batch_data_pack):
        for callback in self._callbacks:
            callback.on_batch_data_pack(batch_data_pack)

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        for callback in self._callbacks:
            callback.on_batch_unpacked(x, y)

    def get_index_pool(self):
        """
        Set the:attr:`_index_pool`.

        Here the :attr:`_index_pool` records the index of all the instances.
        """
        if self._mode == 'point':
            num_instances = len(self._data_pack)
            index_pool = np.expand_dims(range(num_instances), axis=1).tolist()
            return index_pool
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
            return index_pool
        elif self._mode == 'list':
            raise NotImplementedError(
                f'{self._mode} data generator not implemented.')
        else:
            raise ValueError(f"{self._mode} is not a valid mode type"
                             f"Must be one of `point`, `pair` or `list`.")

    def sample(self):
        """Resample the instances from data pack."""
        if self._mode == 'pair':
            self._data_pack.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=self._num_dup,
                num_neg=self._num_neg
            )
        self._index_pool = self.get_index_pool()

    def shuffle(self):
        """Shuffle the instances."""
        np.random.shuffle(self._index_pool)

    def sort(self):
        """Sort the instances by length_right."""
        old_index_pool = self._index_pool
        max_instance_right_length = []
        for row in range(len(old_index_pool)):
            instance = self._data_pack[old_index_pool[row]].unpack()[0]
            max_instance_right_length.append(max(instance['length_right']))
        sort_index = np.argsort(max_instance_right_length)

        self._index_pool = [old_index_pool[index] for index in sort_index]

    @property
    def data_pack(self):
        """`data_pack` getter."""
        return self._data_pack

    @data_pack.setter
    def data_pack(self, value):
        """`data_pack` setter."""
        self._data_pack = value

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

    @property
    def num_dup(self):
        """`num_dup` getter."""
        return self._num_dup

    @num_dup.setter
    def num_dup(self, value):
        """`num_dup` setter."""
        self._num_dup = value

    @property
    def mode(self):
        """`mode` getter."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """`mode` setter."""
        self._mode = value

    @property
    def index_pool(self):
        """`index_pool` getter."""
        return self._index_pool

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
