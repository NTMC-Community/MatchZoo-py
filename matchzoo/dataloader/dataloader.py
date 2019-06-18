""""Basic data loader."""
import typing

import math
import random
import numpy as np
import torch
from torch.utils import data

from matchzoo.dataloader.callbacks import Callback


class DataLoader(data.DataLoader):
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
    :param callbacks: Callbacks. See `matchzoo.dataloader.callbacks` for more
        details.

    Examples:
        >>> import matchzoo as mz
        >>> data_pack = mz.datasets.toy.load_data(stage='train')
        >>> preprocessor = mz.preprocessors.CDSSMPreprocessor()
        >>> data_processed = preprocessor.fit_transform(data_pack)
        >>> dataset = mz.dataloader.Dataset(data_processed, mode='point')
        >>> padding_callback = mz.dataloader.callbacks.CDSSMPadding()
        >>> dataloader = mz.dataloader.DataLoader(
        ...     dataset, stage='train', callbacks=[padding_callback])
        >>> len(dataloader)
        4
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        device: typing.Optional[torch.device] = None,
        stage='train',
        resample: bool = True,
        shuffle: bool = False,
        sort: bool = True,
        callbacks: typing.List[Callback] = None
    ):
        """"Init."""
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type."
                             f"Must be one of `train`, `dev`, `test`.")

        if device is None or not isinstance(device, torch.device):
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        if callbacks is None:
            callbacks = []

        self._dataset = dataset
        self._batch_size = batch_size
        self._device = device
        self._stage = stage
        self._shuffle = shuffle
        self._resample = resample
        self._sort = sort
        self._callbacks = callbacks

        self._batch_indices = 0

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(len(self._dataset) / self._batch_size)

    @property
    def id_left(self) -> np.ndarray:
        x, _ = self._dataset[:]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        _, y = self._dataset[:]
        return y.squeeze() if y is not None else None

    def init_epoch(self):
        """Resample, shuffle or sort the dataset for a new epoch."""
        if self._resample:
            self._dataset.sample()

        if self._sort:
            self._dataset.sort()
        elif self._shuffle:
            self._dataset.shuffle()

        self._batch_indices = 0

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        self.init_epoch()
        while self._batch_indices < len(self):
            low = self._batch_indices * self._batch_size
            high = min(
                (self._batch_indices + 1) * self._batch_size, len(self._dataset))
            batch = self._dataset[low:high]
            self._batch_indices += 1

            x, y = batch
            batch_x = {}
            self._handle_callbacks_on_batch_unpacked(x, y)
            for key, value in x.items():
                if key == 'id_left' or key == 'id_right':
                    continue
                batch_x[key] = torch.Tensor(value.tolist()).to(self._device)

            if self._stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':
                    batch_y = torch.LongTensor(y.squeeze()).to(self._device)
                else:
                    batch_y = torch.FloatTensor(y.squeeze()).to(self._device)
                yield batch_x, batch_y

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        for callback in self._callbacks:
            callback.on_batch_unpacked(x, y)
