import typing

import torch
from torch import nn
from torch import optim

import matchzoo
from matchzoo.engine.base_metric import (
    BaseMetric, RankingMetric, ClassificationMetric
)

activation = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['hardtanh', nn.Hardtanh()],
    ['relu6', nn.ReLU6()],
    ['sigmoid', nn.Sigmoid()],
    ['tanh', nn.Tanh()],
    ['softmax', nn.Softmax()],
    ['softmax2d', nn.Softmax2d()],
    ['logsoftmax', nn.LogSoftmax()],
    ['elu', nn.ELU()],
    ['selu', nn.SELU()],
    ['celu', nn.CELU()],
    ['hardshrink', nn.Hardshrink()],
    ['leakyrelu', nn.LeakyReLU()],
    ['logsigmoid', nn.LogSigmoid()],
    ['softplus', nn.Softplus()],
    ['softshrink', nn.Softshrink()],
    ['prelu', nn.PReLU()],
    ['softsign', nn.Softsign()],
    ['softmin', nn.Softmin()],
    ['tanhshrink', nn.Tanhshrink()],
    ['rrelu', nn.RReLU()],
    ['glu', nn.GLU()],
])

loss = nn.ModuleDict([
    ['l1', nn.L1Loss()],
    ['nll', nn.NLLLoss()],
    ['kldiv', nn.KLDivLoss()],
    ['mse', nn.MSELoss()],
    ['bce', nn.BCELoss()],
    ['bce_with_logits', nn.BCEWithLogitsLoss()],
    ['cosine_embedding', nn.CosineEmbeddingLoss()],
    ['ctc', nn.CTCLoss()],
    ['hinge_embedding', nn.HingeEmbeddingLoss()],
    ['margin_ranking', nn.MarginRankingLoss()],
    ['multi_label_margin', nn.MultiLabelMarginLoss()],
    ['multi_label_soft_margin', nn.MultiLabelSoftMarginLoss()],
    ['multi_margin', nn.MultiMarginLoss()],
    ['smooth_l1', nn.SmoothL1Loss()],
    ['soft_margin', nn.SoftMarginLoss()],
    ['cross_entropy', nn.CrossEntropyLoss()],
    ['triplet_margin', nn.TripletMarginLoss()],
    ['poisson_nll', nn.PoissonNLLLoss()]
])

optimizer = dict({
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'sparse_adam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sgd': optim.SGD
})


def _parse(
    identifier: typing.Union[str, typing.Type[nn.Module], nn.Module],
    dictionary: nn.ModuleDict,
    target: str
) -> nn.Module:
    """
    Parse loss and activation.

    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :param dictionary: nn.ModuleDict instance. Map string identifier to
        nn.Module instance.
    :return: A :class:`nn.Module` instance
    """
    if isinstance(identifier, str):
        if identifier in dictionary:
            return dictionary[identifier]
        else:
            raise ValueError(
                f'Could not interpret {target} identifier: ' + str(identifier)
            )
    elif isinstance(identifier, nn.Module):
        return identifier
    elif issubclass(identifier, nn.Module):
        return identifier()
    else:
        raise ValueError(
            f'Could not interpret {target} identifier: ' + str(identifier)
        )


def parse_activation(
    identifier: typing.Union[str, typing.Type[nn.Module], nn.Module]
) -> nn.Module:
    """
    Retrieves a torch Module instance.

    :param identifier: activation identifier, one of
            - String: name of a activation
            - Torch Modele subclass
            - Torch Module instance (it will be returned unchanged).
    :return: A :class:`nn.Module` instance

    Examples::
        >>> from torch import nn
        >>> from matchzoo.utils import parse_activation

    Use `str` as activation:
        >>> activation = parse_activation('relu')
        >>> type(activation)
        <class 'torch.nn.modules.activation.ReLU'>

    Use :class:`torch.nn.Module` subclasses as activation:
        >>> type(parse_activation(nn.ReLU))
        <class 'torch.nn.modules.activation.ReLU'>

    Use :class:`torch.nn.Module` instances as activation:
        >>> type(parse_activation(nn.ReLU()))
        <class 'torch.nn.modules.activation.ReLU'>

    """

    return _parse(identifier, activation, 'activation')


def parse_loss(
    identifier: typing.Union[str, typing.Type[nn.Module], nn.Module],
    task: typing.Optional[str] = None
) -> nn.Module:
    """
    Retrieves a torch Module instance.

    :param identifier: loss identifier, one of
            - String: name of a loss
            - Torch Module subclass
            - Torch Module instance (it will be returned unchanged).
    :param task: Task type for determining specific loss.
    :return: A :class:`nn.Module` instance

    Examples::
        >>> from torch import nn
        >>> from matchzoo.utils import parse_loss

    Use `str` as loss:
        >>> loss = parse_loss('mse')
        >>> type(loss)
        <class 'torch.nn.modules.loss.MSELoss'>

    Use :class:`torch.nn.Module` subclasses as loss:
        >>> type(parse_loss(nn.MSELoss))
        <class 'torch.nn.modules.loss.MSELoss'>

    Use :class:`torch.nn.Module` instances as loss:
        >>> type(parse_loss(nn.MSELoss()))
        <class 'torch.nn.modules.loss.MSELoss'>

    """
    return _parse(identifier, loss, 'loss')


def _parse_metric(
    metric: typing.Union[str, typing.Type[BaseMetric], BaseMetric],
    Metrix: typing.Type[BaseMetric]
) -> BaseMetric:
    """
    Parse metric.

    :param metrc: Input metric in any form.
    :param Metrix: Base Metric class. Either
        :class:`matchzoo.engine.base_metric.RankingMetric` or
        :class:`matchzoo.engine.base_metric.ClassificationMetric`.
    :return: A :class:`BaseMetric` instance
    """
    if isinstance(metric, str):
        metric = metric.lower()  # ignore case
        for subclass in Metrix.__subclasses__():
            if metric == subclass.ALIAS or metric in subclass.ALIAS:
                return subclass()
    elif isinstance(metric, Metrix):
        return metric
    elif issubclass(metric, Metrix):
        return metric()
    raise ValueError(f'`{metric}` can not be used in current task.')


def parse_metric(
    metric: typing.Union[str, typing.Type[BaseMetric], BaseMetric],
    task: str
) -> BaseMetric:
    """
    Parse input metric in any form into a :class:`BaseMetric` instance.

    :param metric: Input metric in any form.
    :param task: Task type for determining specific metric.
    :return: A :class:`BaseMetric` instance

    Examples::
        >>> from matchzoo import metrics
        >>> from matchzoo.utils import parse_metric

    Use `str` as MatchZoo metrics:
        >>> mz_metric = parse_metric('map', 'ranking')
        >>> type(mz_metric)
        <class 'matchzoo.metrics.mean_average_precision.MeanAveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` subclasses as MatchZoo metrics:
        >>> type(parse_metric(metrics.AveragePrecision, 'ranking'))
        <class 'matchzoo.metrics.average_precision.AveragePrecision'>

    Use :class:`matchzoo.engine.BaseMetric` instances as MatchZoo metrics:
        >>> type(parse_metric(metrics.AveragePrecision(), 'ranking'))
        <class 'matchzoo.metrics.average_precision.AveragePrecision'>

    """
    if task is None:
        raise ValueError(
            'Should specify one `BaseTask`.'
        )
    if task == 'ranking':
        return _parse_metric(metric, RankingMetric)
    if task == 'classification':
        return _parse_metric(metric, ClassificationMetric)
    else:
        raise ValueError(
            'Should be a Ranking or Classification task.'
        )


def parse_optimizer(
    identifier: typing.Union[str, typing.Type[optim.Optimizer]],
) -> optim.Optimizer:
    """
    Parse input metric in any form into a :class:`Optimizer` class.

    :param optimizer: Input optimizer in any form.
    :return: A :class:`Optimizer` class

    Examples::
        >>> from torch import optim
        >>> from matchzoo.utils import parse_optimizer

    Use `str` as optimizer:
        >>> parse_optimizer('adam')
        <class 'torch.optim.adam.Adam'>

    Use :class:`torch.optim.Optimizer` subclasses as optimizer:
        >>> parse_optimizer(optim.Adam)
        <class 'torch.optim.adam.Adam'>

    """
    if isinstance(identifier, str):
        identifier = identifier.lower()  # ignore case
        if identifier in optimizer:
            return optimizer[identifier]
        else:
            raise ValueError(
                f'Could not interpret optimizer identifier: ' + str(identifier)
            )
    elif issubclass(identifier, optim.Optimizer):
        return identifier
    else:
        raise ValueError(
            f'Could not interpret optimizer identifier: ' + str(identifier)
        )
