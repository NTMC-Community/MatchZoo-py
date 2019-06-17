import math
import random
import typing
import numpy as np

import torch
from torch.utils import data

from matchzoo.dataloader.callbacks import Callback


class DataLoader(data.DataLoader):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 1,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        stage = 'train', 
        resample: bool = True,    
        shuffle: bool = False,
        sort: bool = True,
        callbacks: typing.List[Callback] = None
    ):
        """"Init."""
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type."
                             f"Must be one of `train`, `dev`, `test`.")
        
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

    # number of batches
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
        return y.squeeze()

    def init_epoch(self):
        if self.resample:
            self.dataset.sample()
        
        if self.sort:
            self.dataset.sort()
        elif self.shuffle:
            self.dataset.shuffle()

        self.batch_indices = 0
        
    def __iter__(self) -> typing.Tuple[dict, np.ndarray]:
        self.init_epoch()
        while self.batch_indices < len(self):
            low = self.batch_indices * self.batch_size
            high = min((self.batch_indices + 1) * self.batch_size, len(self.dataset))
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