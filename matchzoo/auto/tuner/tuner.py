import copy
import typing
import logging

import torch
import hyperopt
import numpy as np

import matchzoo as mz
from matchzoo.engine.base_metric import BaseMetric
from matchzoo.utils import parse_optimizer


class Tuner(object):
    """
    Model hyper-parameters tuner.

    `model.params.hyper_space` reprensents the model's hyper-parameters
    search space, which is the cross-product of individual hyper parameter's
    hyper space. When a `Tuner` builds a model, for each hyper parameter in
    `model.params`, if the hyper-parameter has a hyper-space, then a sample
    will be taken in the space. However, if the hyper-parameter does not
    have a hyper-space, then the default value of the hyper-parameter will
    be used.

    See `tutorials/model_tuning.ipynb` for a detailed walkthrough on usage.

    :param params: A completed parameter table to tune. Usually `model.params`
        of the desired model to tune. `params.completed()` should be `True`.
    :param optimizer: Str or `Optimizer` class. Optimizer for optimizing model.
    :param trainloader: Training data to use. Should be a `DataLoader`.
    :param validloader: Testing data to use. Should be a `DataLoader`.
    :param embedding: Embedding used by model.
    :param fit_kwargs: Extra keyword arguments to pass to `fit`.
        (default: `dict(epochs=10, verbose=0)`)
    :param metric: Metric to tune upon. Must be one of the metrics in
        `model.params['task'].metrics`. (default: the first metric in
        `params.['task'].metrics`.
    :param mode: Either `maximize` the metric or `minimize` the metric.
        (default: 'maximize')
    :param num_runs: Number of runs. Each run takes a sample in
        `params.hyper_space` and build a model based on the sample.
        (default: 10)
    :param verbose: Verbosity. (default: 1)

    """

    def __init__(
        self,
        params: 'mz.ParamTable',
        optimizer: str = 'adam',
        trainloader: mz.dataloader.DataLoader = None,
        validloader: mz.dataloader.DataLoader = None,
        embedding: np.ndarray = None,
        fit_kwargs: dict = None,
        metric: typing.Union[str, BaseMetric] = None,
        mode: str = 'maximize',
        num_runs: int = 10,
        verbose=1
    ):
        """Tuner."""
        if fit_kwargs is None:
            fit_kwargs = dict(epochs=5, verbose=0)

        if 'with_embedding' in params:
            params['embedding'] = embedding
            params['embedding_input_dim'] = embedding.shape[0]
            params['embedding_output_dim'] = embedding.shape[1]
        self._validate_params(params)

        metric = metric or params['task'].metrics[0]
        self._validate_optimizer(optimizer)
        self._validate_dataloader(trainloader)
        self._validate_dataloader(validloader)
        self._validate_kwargs(fit_kwargs)
        self._validate_mode(mode)
        self._validate_metric(params, metric)

        self.__curr_run_num = 0

        # these variables should not change within the same `tune` call
        self._params = params
        self._optimizer = parse_optimizer(optimizer)
        self._trainloader = trainloader
        self._validloader = validloader
        self._embedding = embedding
        self._fit_kwargs = fit_kwargs
        self._metric = metric
        self._mode = mode
        self._num_runs = num_runs
        self._verbose = verbose

    def tune(self):
        """
        Start tuning.

        Notice that `tune` does not affect the tuner's inner state, so each
        new call to `tune` starts fresh. In other words, hyperspaces are
        suggestive only within the same `tune` call.
        """
        if self.__curr_run_num != 0:
            print(
                """WARNING: `tune` does not affect the tuner's inner state, so
                each new call to `tune` starts fresh. In other words,
                hyperspaces are suggestive only within the same `tune` call."""
            )
        self.__curr_run_num = 0
        logging.getLogger('hyperopt').setLevel(logging.CRITICAL)

        trials = hyperopt.Trials()

        self._fmin(trials)

        return {
            'best': trials.best_trial['result']['mz_result'],
            'trials': [trial['result']['mz_result'] for trial in trials.trials]
        }

    def _fmin(self, trials):
        # new version of hyperopt has keyword argument `show_progressbar` that
        # breaks doctests, so here's a workaround
        fmin_kwargs = dict(
            fn=self._run,
            space=self._params.hyper_space,
            algo=hyperopt.tpe.suggest,
            max_evals=self._num_runs,
            trials=trials
        )
        try:
            hyperopt.fmin(
                **fmin_kwargs,
                show_progressbar=False
            )
        except TypeError:
            hyperopt.fmin(**fmin_kwargs)

    def _run(self, sample):
        self.__curr_run_num += 1

        # build model
        params = self._create_full_params(sample)
        model = params['model_class'](params=params)
        model.build()

        trainer = mz.trainers.Trainer(
            model=model,
            optimizer=self._optimizer(model.parameters()),
            trainloader=self._trainloader,
            validloader=self._validloader,
            **self._fit_kwargs,
        )

        # fit & evaluate
        trainer.run()

        lookup = trainer.evaluate(self._validloader)
        score = lookup[self._metric]

        # collect result
        # this result is for users, visible outside
        mz_result = {
            '#': self.__curr_run_num,
            'params': params,
            'sample': sample,
            'score': score
        }

        if self._verbose:
            self._log_result(mz_result)

        return {
            # these two items are for hyperopt
            'loss': self._fix_loss_sign(score),
            'status': hyperopt.STATUS_OK,

            # this item is for storing matchzoo information
            'mz_result': mz_result
        }

    def _create_full_params(self, sample):
        params = copy.deepcopy(self._params)
        params.update(sample)
        return params

    def _fix_loss_sign(self, loss):
        if self._mode == 'maximize':
            loss = -loss
        return loss

    @classmethod
    def _log_result(cls, result):
        print(f"Run #{result['#']}")
        print(f"Score: {result['score']}")
        print(result['params'])
        print()

    @property
    def params(self):
        """`params` getter."""
        return self._params

    @params.setter
    def params(self, value):
        """`params` setter."""
        self._validate_params(value)
        self._validate_metric(value, self._metric)
        self._params = value

    @property
    def trainloader(self):
        """`trainloader` getter."""
        return self._trainloader

    @trainloader.setter
    def trainloader(self, value):
        """`trainloader` setter."""
        self._validate_dataloader(value)
        self._trainloader = value

    @property
    def validloader(self):
        """`validloader` getter."""
        return self._validloader

    @validloader.setter
    def validloader(self, value):
        """`validloader` setter."""
        self._validate_dataloader(value)
        self._validloader = value

    @property
    def fit_kwargs(self):
        """`fit_kwargs` getter."""
        return self._fit_kwargs

    @fit_kwargs.setter
    def fit_kwargs(self, value):
        """`fit_kwargs` setter."""
        self._validate_kwargs(value)
        self._fit_kwargs = value

    @property
    def metric(self):
        """`metric` getter."""
        return self._metric

    @metric.setter
    def metric(self, value):
        """`metric` setter."""
        self._validate_metric(self._params, value)
        self._metric = value

    @property
    def mode(self):
        """`mode` getter."""
        return self._mode

    @mode.setter
    def mode(self, value):
        """`mode` setter."""
        self._validate_mode(value)
        self._mode = value

    @property
    def num_runs(self):
        """`num_runs` getter."""
        return self._num_runs

    @num_runs.setter
    def num_runs(self, value):
        """`num_runs` setter."""
        self._validate_num_runs(value)
        self._num_runs = value

    @property
    def verbose(self):
        """`verbose` getter."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """`verbose` setter."""
        self._verbose = value

    @classmethod
    def _validate_params(cls, params):
        if not isinstance(params, mz.ParamTable):
            raise TypeError("Only accepts a `ParamTable` instance.")
        if not params.hyper_space:
            raise ValueError("Parameter hyper-space empty.")
        if not params.completed(exclude=['out_activation_func']):
            raise ValueError("Parameters not complete.")

    @classmethod
    def _validate_optimizer(cls, optimizer):
        if not isinstance(optimizer, (str, torch.optim.Optimizer)):
            raise TypeError(
                "Only accepts a `Optimizer` instance.")

    @classmethod
    def _validate_dataloader(cls, data):
        if not isinstance(data, mz.dataloader.DataLoader):
            raise TypeError(
                "Only accepts a `DataLoader` instance.")

    @classmethod
    def _validate_kwargs(cls, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError('Only accepts a `dict` instance.')

    @classmethod
    def _validate_mode(cls, mode):
        if mode not in ('maximize', 'minimize'):
            raise ValueError('`mode` should be one of `maximize`, `minimize`.')

    @classmethod
    def _validate_metric(cls, params, metric):
        if metric not in params['task'].metrics:
            raise ValueError('Target metric does not exist in the task.')

    @classmethod
    def _validate_num_runs(cls, num_runs):
        if not isinstance(num_runs, int):
            raise TypeError('Only accepts an `int` value.')
