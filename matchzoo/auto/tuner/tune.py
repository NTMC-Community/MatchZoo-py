import typing

import numpy as np

import matchzoo as mz
from matchzoo.engine.base_metric import BaseMetric
from .tuner import Tuner


def tune(
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
    """
    Tune model hyper-parameters.

    A simple shorthand for using :class:`matchzoo.auto.Tuner`.

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
    :param callbacks: A list of callbacks to handle. Handled sequentially
        at every callback point.
    :param verbose: Verbosity. (default: 1)

    Example:
        >>> import matchzoo as mz
        >>> import numpy as np
        >>> train = mz.datasets.toy.load_data('train')
        >>> valid = mz.datasets.toy.load_data('dev')
        >>> prpr = mz.models.DenseBaseline.get_default_preprocessor()
        >>> train = prpr.fit_transform(train, verbose=0)
        >>> valid = prpr.transform(valid, verbose=0)
        >>> trainset = mz.dataloader.Dataset(train)
        >>> validset = mz.dataloader.Dataset(valid)
        >>> padding = mz.models.DenseBaseline.get_default_padding_callback()
        >>> trainloader = mz.dataloader.DataLoader(trainset, callback=padding)
        >>> validloader = mz.dataloader.DataLoader(validset, callback=padding)
        >>> model = mz.models.DenseBaseline()
        >>> model.params['task'] = mz.tasks.Ranking()
        >>> optimizer = 'adam'
        >>> embedding = np.random.uniform(-0.2, 0.2,
        ...     (prpr.context['vocab_size'], 100))
        >>> tuner = mz.auto.Tuner(
        ...     params=model.params,
        ...     optimizer=optimizer,
        ...     trainloader=trainloader,
        ...     validloader=validloader,
        ...     embedding=embedding,
        ...     num_runs=1,
        ...     verbose=0
        ... )
        >>> results = tuner.tune()
        >>> sorted(results['best'].keys())
        ['#', 'params', 'sample', 'score']

    """

    tuner = Tuner(
        params=params,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        embedding=embedding,
        fit_kwargs=fit_kwargs,
        metric=metric,
        mode=mode,
        num_runs=num_runs,
        verbose=verbose
    )
    return tuner.tune()
