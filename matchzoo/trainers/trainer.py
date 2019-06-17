"""Base Trainer."""

import typing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matchzoo
from matchzoo import tasks
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
        if None, uses the current device for the default tensor type
        (see torch.set_default_tensor_type()). device will be the CPU
        for CPU tensor types and the current CUDA device for CUDA
        tensor types.
    :param start_epoch: Int. Number of starting epoch.
    :param epochs: The maximum number of epochs for training.
        Defaults to 10.
    :param validate_interval: Int. Interval of validation.
    :param scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    :param patience: Number fo events to wait if no improvement and
        then stop the training.
    :param data_parallel: Bool. Whether support data parallel.
    :param checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    :param save_path: Path to save trainer.
    :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
        device: typing.Optional[torch.device] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        scheduler: typing.Any = None,
        patience: typing.Optional[int] = None,
        data_parallel: bool = True,
        checkpoint: typing.Union[str, typing.Any] = None,
        save_path: typing.Optional[str] = None,
        verbose: int = 1,
        **kwargs
    ):
        """Base Trainer constructor."""
        self._load_model(model, device, data_parallel)
        self._load_dataloader(
            trainloader, validloader, validate_interval
        )

        self._task = self._model.params['task']
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._criterions = self._task.losses
        self._early_stopping = EarlyStopping(
            patience=patience,
            key=self._task.metrics[0]
        )

        self._start_epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0

        self._verbose = verbose

        if checkpoint:
            self.restore(checkpoint)
        # TODO: Change Path
        if save_path:
            self._save_path = save_path
        else:
            self._save_path = './save'

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
        :param validate_interval: int. Interval of validation.
        """
        if not isinstance(trainloader, DataLoader):
            raise ValueError(
                'trainloader should be a `DataLoader` instance.'
            )
        if not isinstance(trainloader, DataLoader):
            raise ValueError(
                'validloader should be a `DataLoader` instance.'
            )
        self.trainloader = trainloader
        self.validloader = validloader
        if not validate_interval:
            self.validate_interval = len(self.trainloader)
        else:
            self.validate_interval = validate_interval

    def _load_model(
        self,
        model: BaseModel,
        device: typing.Optional[torch.device],
        data_parallel: bool = True
    ):
        """
        Load model.

        :param model: :class:`BaseModel` instance.
        :param device: the desired device of returned tensor.
            Default: if None, uses the current device for the
            default tensor type (see torch.set_default_tensor_type()).
            device will be the CPU for CPU tensor types and the
            current CUDA device for CUDA tensor types.
        :param data_parallel: bool. Whether support data parallel.
        """
        if not isinstance(model, BaseModel):
            raise ValueError(
                'model should be a `BaseModel` instance.'
                f' But got {type(model)}.'
            )
        self.device = device
        self._model = model.to(self.device)
        if (("cuda" in str(self.device)) and (
                torch.cuda.device_count() > 1) and (
                    data_parallel is True)):
            self._model = torch.nn.DataParallel(self._model)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.
        """
        self._optimizer.zero_grad()
        loss.backward()
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
        num_batch = len(self.trainloader)
        train_loss = AverageMeter()
        with tqdm(enumerate(self.trainloader), total=num_batch) as pbar:
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
                if self._iteration % self.validate_interval == 0:
                    pbar.update()
                    pbar.write(
                        f'[Iter-{self._iteration} '
                        f'Loss-{train_loss.avg:.3f}]:')
                    result = self.evaluate(self.validloader)
                    if self._verbose:
                        pbar.write('  Validation: ' + ' - '.join(
                            f'{k}: {round(v, 4)}' for k, v in result.items()))
                    # Early stopping
                    self._early_stopping.update(result)
                    if self._early_stopping.should_stop_early:
                        self.save()
                        pbar.write('Ran out of patience. Stop training...')
                        break
                    elif self._early_stopping.is_best_so_far:
                        self.save()

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
                    metric, id_left, y_true, y_pred.squeeze())
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

    def save(self):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        """
        best_so_far = self._early_stopping.best_so_far
        save_path = Path(self._save_path).joinpath(
            f'model_epoch-{self._epoch}_best-{best_so_far:.4f}.pt'
        )
        checkpoint = {
            'epoch': self._epoch,
            'best_so_far': best_so_far,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'early_stopping': self._optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    # TODO: model save
    def restore(self, checkpoint: typing.Union[str, typing.Any]):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.
            If None, training starts from scratch. Defaults to None.
            Should be a file-like object (has to implement read, readline,
            tell, and seek), or a string containing a file name.

        """
        checkpoint = torch.load(checkpoint)
        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._start_epoch = checkpoint['epoch']
        self._early_stopping.load_state_dict(
            checkpoint['early_stopping'])
