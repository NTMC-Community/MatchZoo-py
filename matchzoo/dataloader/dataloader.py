"""Basic data loader."""
import typing
import math

import numpy as np
import torch
from torch.utils import data

from matchzoo.dataloader.dataset import Dataset
from matchzoo.engine.base_callback import BaseCallback


class DataLoader(object):
    """
    DataLoader that loads batches of data from a Dataset.

    :param dataset: The Dataset object to load data from.
    :param device: The desired device of returned tensor. Default: if None,
        use the current device. If `torch.device` or int, use device specified
        by user. If list, the first item will be used.
    :param stage: One of "train", "dev", and "test". (default: "train")
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
        >>> preprocessor = mz.preprocessors.BasicPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset = mz.dataloader.Dataset(
        ...     data_processed, mode='point', batch_size=32)
        >>> padding_callback = mz.dataloader.callbacks.BasicPadding()
        >>> dataloader = mz.dataloader.DataLoader(
        ...     dataset, stage='train', callback=padding_callback)
        >>> len(dataloader)
        4

    """

    def __init__(
        self,
        dataset: Dataset,
        device: typing.Union[torch.device, int, list, None] = None,
        stage='train',
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

        if isinstance(device, list) and len(device):
            device = device[0]
        elif not (isinstance(device, torch.device) or isinstance(device, int)):
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._dataset = dataset
        self._pin_momory = pin_memory
        self._timeout = timeout
        self._num_workers = num_workers
        self._worker_init_fn = worker_init_fn
        self._device = device
        self._stage = stage
        self._callback = callback

        self._dataloader = data.DataLoader(
            self._dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=lambda x: x,
            batch_sampler=None,
            num_workers=self._num_workers,
            pin_memory=self._pin_momory,
            timeout=self._timeout,
            worker_init_fn=self._worker_init_fn,
        )

    def __len__(self) -> int:
        """Get the total number of batches."""
        return len(self._dataset)

    @property
    def id_left(self) -> np.ndarray:
        """`id_left` getter."""
        x, _ = self._dataset[:]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        """`label` getter."""
        _, y = self._dataset[:]
        return y.squeeze() if y is not None else None

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        """Iteration."""
        for batch_data in self._dataloader:
            x, y = batch_data
            self._handle_callbacks_on_batch_unpacked(x, y)

            batch_x = {}
            for key, value in x.items():
                if key == 'id_left' or key == 'id_right':
                    continue
                batch_x[key] = torch.tensor(
                    value, device=self._device)

            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':  # task='classification'
                    batch_y = torch.tensor(
                        y.squeeze(axis=-1), dtype=torch.long, device=self._device)
                else:  # task='ranking'
                    batch_y = torch.tensor(
                        y, dtype=torch.float, device=self._device)
                yield batch_x, batch_y

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        if self._callback is not None:
            self._callback.on_batch_unpacked(x, y)
