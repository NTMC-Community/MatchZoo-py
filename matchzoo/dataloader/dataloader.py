""""Basic data loader."""
import typing

import math
import random
import warnings
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

        if device and not isinstance(device, torch.device):
            warnings.warn('The `device` argument should be a `torch.device` '
                          'instance. Currently it will be set according to '
                          'this device.')
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        elif device is None:
            warnings.warn('The `device` argument has been set according to '
                          'this device.')
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        if shuffle and sort:
            warnings.warn('The `sort` argument has been set to True, so '
                          'shuffle=True will be invalid.')

        if callbacks is None:
            callbacks = []

        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.stage = stage
        self.shuffle = shuffle
        self.resample = resample
        self.sort = sort
        self.callbacks = callbacks

        self.batch_indices = 0

    def __len__(self) -> int:
        """Get the total number of batches."""
        return math.ceil(len(self.dataset) / self.batch_size)

    @property
    def id_left(self) -> np.ndarray:
        x, _ = self.dataset[:]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        _, y = self.dataset[:]
        return y.squeeze() if y is not None else None

    def init_epoch(self):
        """Resample, shuffle or sort the dataset for a new epoch."""
        if self.resample:
            self.dataset.sample()

        if self.sort:
            self.dataset.sort()
        elif self.shuffle:
            self.dataset.shuffle()

        self.batch_indices = 0

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        self.init_epoch()
        while self.batch_indices < len(self):
            low = self.batch_indices * self.batch_size
            high = min(
                (self.batch_indices + 1) * self.batch_size, len(self.dataset))
            batch = self.dataset[low:high]
            self.batch_indices += 1

            x, y = batch
            batch_x = {}
            self._handle_callbacks_on_batch_unpacked(x, y)
            for key, value in x.items():
                if key == 'id_left' or key == 'id_right':
                    continue
                batch_x[key] = torch.Tensor(value.tolist()).to(self.device)

            if self.stage == 'test':
                yield batch_x, None
            else:
                if y.dtype == 'int':
                    batch_y = torch.LongTensor(y.squeeze()).to(self.device)
                else:
                    batch_y = torch.FloatTensor(y.squeeze()).to(self.device)
                yield batch_x, batch_y

    def _handle_callbacks_on_batch_unpacked(self, x, y):
        for callback in self.callbacks:
            callback.on_batch_unpacked(x, y)
