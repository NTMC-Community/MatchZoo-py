"""Base Trainer."""

import typing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import matchzoo
from matchzoo import tasks
from matchzoo.dataloader import DataLoader
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_metric import BaseMetric
from matchzoo.utils import AverageMeter, Timer, EarlyStopping


class Trainer:
    """
    MatchZoo tranier.

    :param model: A :class:`BaseModel` instance.
    :param optimizer: A :class:`optim.Optimizer` instance.
    :param trainloader: A :class`DataLoader` instance. The dataloader
        is used for training the model.
    :param validloader: A :class`DataLoader` instance. The dataloader
        is used for validating the model.
    :param device: The desired device of returned tensor. Default:
        if None, use the current device. If `torch.device` or int,
        use device specified by user. If list, use data parallel.
    :param start_epoch: Int. Number of starting epoch.
    :param epochs: The maximum number of epochs for training.
        Defaults to 10.
    :param validate_interval: Int. Interval of validation.
    :param scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    :param clip_norm: Max norm of the gradients to be clipped.
    :param patience: Number fo events to wait if no improvement and
        then stop the training.
    :param key: Key of metric to be compared.
    :param checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    :param save_dir: Directory to save trainer.
    :param save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.
    :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        device: typing.Union[torch.device, int, list, None] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        patience: typing.Optional[int] = None,
        key: typing.Any = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,
        **kwargs
    ):
        """Base Trainer constructor."""
        self._load_model(model, device)
        self._load_dataloader(
            trainloader, validloader, validate_interval
        )

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._clip_norm = clip_norm
        self._criterions = self._task.losses

        if not key:
            key = self._task.metrics[0]
        self._early_stopping = EarlyStopping(
            patience=patience,
            key=key
        )

        self._start_epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0
        self._verbose = verbose
        self._save_all = save_all

        self._load_path(checkpoint, save_dir)

    def _load_dataloader(
        self,
        trainloader: DataLoader,
        validloader: DataLoader,
        validate_interval: typing.Optional[int] = None
    ):
        """
        Load trainloader and determine validate interval.

        :param trainloader: A :class`DataLoader` instance. The dataloader
            is used to train the model.
        :param validloader: A :class`DataLoader` instance. The dataloader
            is used to validate the model.
        :param validate_interval: int. Interval of validation.
        """
        if not isinstance(trainloader, DataLoader):
            raise ValueError(
                'trainloader should be a `DataLoader` instance.'
            )
        if not isinstance(validloader, DataLoader):
            raise ValueError(
                'validloader should be a `DataLoader` instance.'
            )
        self._trainloader = trainloader
        self._validloader = validloader
        if not validate_interval:
            self._validate_interval = len(self._trainloader)
        else:
            self._validate_interval = validate_interval

    def _load_model(
        self,
        model: BaseModel,
        device: typing.Union[torch.device, int, list, None] = None
    ):
        """
        Load model.

        :param model: :class:`BaseModel` instance.
        :param device: The desired device of returned tensor. Default:
            if None, use the current device. If `torch.device` or int,
            use device specified by user. If list, use data parallel.
        """
        if not isinstance(model, BaseModel):
            raise ValueError(
                'model should be a `BaseModel` instance.'
                f' But got {type(model)}.'
            )

        self._task = model.params['task']
        self._data_parallel = False
        self._model = model

        if isinstance(device, list) and len(device):
            self._data_parallel = True
            self._model = torch.nn.DataParallel(self._model, device_ids=device)
            self._device = device[0]
        else:
            if not (isinstance(device, torch.device) or isinstance(device, int)):
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

        if isinstance(self._device, int):
            self._device = torch.device(f"cuda:{self._device}")

        self._model.to(self._device)

    def _load_path(
        self,
        checkpoint: typing.Union[str, Path],
        save_dir: typing.Union[str, Path],
    ):
        """
        Load save_dir and Restore from checkpoint.

        :param checkpoint: A checkpoint from which to continue training.
            If None, training starts from scratch. Defaults to None.
            Should be a file-like object (has to implement read, readline,
            tell, and seek), or a string containing a file name.
        :param save_dir: Directory to save trainer.

        """
        if not save_dir:
            save_dir = Path('.').joinpath('save')
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)

        self._save_dir = Path(save_dir)
        # Restore from checkpoint

        if checkpoint:
            if self._save_all:
                self.restore(checkpoint)
            else:
                self.restore_model(checkpoint)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.

        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_norm:
            nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_norm
            )
        self._optimizer.step()

    def _run_scheduler(self):
        """Run scheduler."""
        if self._scheduler:
            self._scheduler.step()

    def run(self):
        """
        Train model.

        The processes:
            Run each epoch -> Run scheduler -> Should stop early?

        """
        self._model.train()
        timer = Timer()
        for epoch in range(self._start_epoch, self._epochs + 1):
            self._epoch = epoch
            self._run_epoch()
            self._run_scheduler()
            if self._early_stopping.should_stop_early:
                break
        if self._verbose:
            tqdm.write(f'Cost time: {timer.time}s')

    def _run_epoch(self):
        """
        Run each epoch.

        The training steps:
            - Get batch and feed them into model
            - Get outputs. Caculate all losses and sum them up
            - Loss backwards and optimizer steps
            - Evaluation
            - Update and output result

        """
        # Get total number of batch
        num_batch = len(self._trainloader)
        train_loss = AverageMeter()
        with tqdm(enumerate(self._trainloader), total=num_batch,
                  disable=not self._verbose) as pbar:
            for step, (inputs, target) in pbar:
                outputs = self._model(inputs)
                # Caculate all losses and sum them up
                loss = torch.sum(
                    *[c(outputs, target) for c in self._criterions]
                )
                self._backward(loss)
                train_loss.update(loss.item())

                # Set progress bar
                pbar.set_description(f'Epoch {self._epoch}/{self._epochs}')
                pbar.set_postfix(loss=f'{loss.item():.3f}')

                # Run validate
                self._iteration += 1
                if self._iteration % self._validate_interval == 0:
                    pbar.update(1)
                    if self._verbose:
                        pbar.write(
                            f'[Iter-{self._iteration} '
                            f'Loss-{train_loss.avg:.3f}]:')
                    result = self.evaluate(self._validloader)
                    if self._verbose:
                        pbar.write('  Validation: ' + ' - '.join(
                            f'{k}: {round(v, 4)}' for k, v in result.items()))
                    # Early stopping
                    self._early_stopping.update(result)
                    if self._early_stopping.is_best_so_far:
                        self._save()
                    if self._early_stopping.should_stop_early:
                        pbar.write('Ran out of patience. Stop training...')
                        break                     
 
    def evaluate(
        self,
        dataloader: DataLoader,
    ):
        """
        Evaluate the model.

        :param dataloader: A DataLoader object to iterate over the data.

        """
        result = dict()
        y_pred = self.predict(dataloader)
        y_true = dataloader.label
        id_left = dataloader.id_left

        if isinstance(self._task, tasks.Classification):
            for metric in self._task.metrics:
                result[metric] = metric(y_true, y_pred)
        else:
            for metric in self._task.metrics:
                result[metric] = self._eval_metric_on_data_frame(
                    metric, id_left, y_true, y_pred.squeeze(axis=-1))
        return result

    @classmethod
    def _eval_metric_on_data_frame(
        cls,
        metric: BaseMetric,
        id_left: typing.Any,
        y_true: typing.Union[list, np.array],
        y_pred: typing.Union[list, np.array]
    ):
        """
        Eval metric on data frame.

        This function is used to eval metrics for `Ranking` task.

        :param metric: Metric for `Ranking` task.
        :param id_left: id of input left. Samples with same id_left should
            be grouped for evaluation.
        :param y_true: Labels of dataset.
        :param y_pred: Outputs of model.
        :return: Evaluation result.

        """
        eval_df = pd.DataFrame(data={
            'id': id_left,
            'true': y_true,
            'pred': y_pred
        })
        assert isinstance(metric, BaseMetric)
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

    def predict(
        self,
        dataloader: DataLoader
    ) -> np.array:
        """
        Generate output predictions for the input samples.

        :param dataloader: input DataLoader
        :return: predictions

        """
        with torch.no_grad():
            self._model.eval()
            predictions = []
            for batch in dataloader:
                inputs = batch[0]
                outputs = self._model(inputs).detach().cpu()
                predictions.append(outputs)
            self._model.train()
            return torch.cat(predictions, dim=0).numpy()

    def _save(self):
        """Save."""
        if self._save_all:
            self.save()
        else:
            self.save_model()

    def save_model(self):
        """Save the model."""
        checkpoint = self._save_dir.joinpath('model.pt')
        if self._data_parallel:
            torch.save(self._model.module.state_dict(), checkpoint)
        else:
            torch.save(self._model.state_dict(), checkpoint)

    def save(self):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        :param path: Path to save trainer.

        """
        checkpoint = self._save_dir.joinpath('trainer.pt')
        if self._data_parallel:
            model = self._model.module.state_dict()
        else:
            model = self._model.state_dict()
        state = {
            'epoch': self._epoch,
            'model': model,
            'optimizer': self._optimizer.state_dict(),
            'early_stopping': self._early_stopping.state_dict(),
        }
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()
        torch.save(state, checkpoint)

    def restore_model(self, checkpoint: typing.Union[str, Path]):
        """
        Restore model.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state)
        else:
            self._model.load_state_dict(state)

    def restore(self, checkpoint: typing.Union[str, Path] = None):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state['model'])
        else:
            self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._start_epoch = state['epoch'] + 1
        self._early_stopping.load_state_dict(state['early_stopping'])
        if self._scheduler:
            self._scheduler.load_state_dict(state['scheduler'])
