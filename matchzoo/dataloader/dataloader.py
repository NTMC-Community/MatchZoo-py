"""Basic data loader."""
import typing

import math
import random
import collections
import numpy as np
import torch
from torch.utils import data

from matchzoo.engine.base_callback import BaseCallback
from matchzoo.dataloader.sampler import (SequentialSampler, RandomSampler,
                                         SortedSampler, BatchSampler)


class DataLoader(object):
    """
    DataLoader that loads batches of data from a Dataset.

    :param dataset: The Dataset object to load data from.
    :param batch_size: Batch_size. (default: 32)
    :param device: An instance of `torch.device` specifying which device
        the Variables are going to be created on.
    :param stage: One of "train", "dev", and "test". (default: "train")
    :param resample: Whether to resample data between epochs. only effective
        when `mode` of dataset is "pair". (default: `True`)
    :param shuffle: Whether to shuffle data between epochs. (default: `False`)
    :param sort: Whether to sort data according to length_right. (default:
        `True`)
    :param callback: BaseCallback. See
        `matchzoo.engine.base_callback.BaseCallback` for more details.
    :param pin_momory: If set to `True`, tensors will be copied into
        pinned memory. (default: `False`)
    :param timeout: The timeout value for collecting a batch from workers. (
        default: 0)
    :param num_workers: The number of subprocesses to use for data loading. 0
        means that the data will be loaded in the main process. (default: 0)
    :param worker_init_fn: If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in [0, num_workers - 1])
        as input, after seeding and before data loading. (default: None)

    Examples:
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_data(stage='train')
        >>> preprocessor = mz.preprocessors.CDSSMPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset = mz.dataloader.Dataset(data_processed, mode='point')
        >>> padding_callback = mz.dataloader.callbacks.CDSSMPadding()
        >>> dataloader = mz.dataloader.DataLoader(
        ...     dataset, stage='train', callback=padding_callback)
        >>> len(dataloader)
        4

    """

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 32,
        device: typing.Optional[torch.device] = None,
        stage='train',
        resample: bool = True,
        shuffle: bool = False,
        sort: bool = True,
        callback: BaseCallback = None,
        pin_memory: bool = False,
        timeout: int = 0,
        num_workers: int = 0,
        worker_init_fn=None,
    ):
        """Init."""
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type."
                             f"Must be one of `train`, `dev`, `test`.")

        if shuffle and sort:
            raise ValueError(f"parameters `shuffle` and `sort` conflict, "
                             f"should not both be `True`.")

        if device is None or not isinstance(device, torch.device):
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._sort = sort
        self._resample = resample

        self._pin_momory = pin_memory
        self._timeout = timeout
        self._num_workers = num_workers
        self._worker_init_fn = worker_init_fn

        self._device = device
        self._stage = stage
        self._callback = callback

        self._dataloader = None

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(len(self._dataset) / self._batch_size)

    @property
    def id_left(self) -> np.ndarray:
        """`id_left` getter."""
        indices = sum(self._dataset.index_pool[:], [])
        x, _ = self._dataset[indices]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        """`label` getter."""
        indices = sum(self._dataset.index_pool[:], [])
        _, y = self._dataset[indices]
        return y.squeeze() if y is not None else None

    def init_epoch(self):
        """Resample, shuffle or sort the dataset for a new epoch."""
        if self._resample:
            self._dataset.sample()

        if not self._shuffle and not self._sort:
            sampler = SequentialSampler(self._dataset)
        elif not self._shuffle and self._sort:
            sampler = SortedSampler(self._dataset)
        elif self._shuffle and not self._sort:
            sampler = RandomSampler(self._dataset)

        batch_sampler = BatchSampler(
            sampler, self._batch_size)

        self._dataloader = data.DataLoader(
            self._dataset,
            collate_fn=mz_collate,
            batch_sampler=batch_sampler,
            num_workers=self._num_workers,
            pin_memory=False,
            timeout=self._timeout,
            worker_init_fn=self._worker_init_fn,
        )

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        """Iteration."""
        self.init_epoch()
        for batch_data in self._dataloader:
            x, y = batch_data
            self._handle_callbacks_on_batch_unpacked(x, y)

            batch_x = {}
            for key, value in x.items():
                if key == 'id_left' or key == 'id_right':
                    continue
                batch_x[key] = torch.tensor(
                    value.tolist(),
                    device=self._device,
                    pin_memory=self._pin_momory)

            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':
                    batch_y = torch.tensor(
                        y.squeeze(axis=-1), dtype=torch.long,
                        device=self._device, pin_memory=self._pin_momory
                    )
                else:
                    batch_y = torch.tensor(
                        y, dtype=torch.float,
                        device=self._device, pin_memory=self._pin_momory
                    )
                yield batch_x, batch_y

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        if self._callback is not None:
            self._callback.on_batch_unpacked(x, y)


def mz_collate(batch):
    """Put each data field into an array with outer dimension batch size."""

    batch_x = collections.defaultdict(list)
    batch_y = []

    for x, y in batch:
        for key in x.keys():
            batch_x[key].append(np.squeeze(x[key], axis=0))
        if y is not None:
            batch_y.append(np.squeeze(y, axis=0))

    for key in batch_x.keys():
        batch_x[key] = np.array(batch_x[key])

    if len(batch_y) == 0:
        batch_y = None
    else:
        batch_y = np.array(batch_y)

    return batch_x, batch_y
