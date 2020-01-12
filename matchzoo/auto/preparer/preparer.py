import typing

import numpy as np

import matchzoo as mz
from matchzoo.engine.base_task import BaseTask
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.base_preprocessor import BasePreprocessor
from matchzoo.dataloader import DatasetBuilder
from matchzoo.dataloader import DataLoaderBuilder


class Preparer(object):
    """
    Unified setup processes of all MatchZoo models.

    `config` is used to control specific behaviors. The default `config`
    will be updated accordingly if a `config` dictionary is passed. e.g. to
    override the default `bin_size`, pass `config={'bin_size': 15}`.

    See `tutorials/automation.ipynb` for a detailed walkthrough on usage.

    Default `config`:

    {
        # pair generator builder kwargs
        'num_dup': 1,

        # histogram unit of DRMM
        'bin_size': 30,
        'hist_mode': 'LCH',

        # dynamic Pooling of MatchPyramid
        'compress_ratio_left': 1.0,
        'compress_ratio_right': 1.0,

        # if no `matchzoo.Embedding` is passed to `tune`
        'embedding_output_dim': 50
    }

    :param task: Task.
    :param config: Configuration of specific behaviors.

    Example:
        >>> import matchzoo as mz
        >>> task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss())
        >>> preparer = mz.auto.Preparer(task)
        >>> model_class = mz.models.DenseBaseline
        >>> train_raw = mz.datasets.toy.load_data('train', 'ranking')
        >>> model, prpr, dsb, dlb = preparer.prepare(model_class,
        ...                                          train_raw)
        >>> model.params.completed(exclude=['out_activation_func'])
        True

    """

    def __init__(
        self,
        task: BaseTask,
        config: typing.Optional[dict] = None
    ):
        """Init."""
        self._task = task
        self._config = self.get_default_config()
        if config:
            self._config.update(config)

        self._infer_num_neg()

    def prepare(
        self,
        model_class: typing.Type[BaseModel],
        data_pack: mz.DataPack,
        callback: typing.Optional[BaseCallback] = None,
        preprocessor: typing.Optional[BasePreprocessor] = None,
        embedding: typing.Optional['mz.Embedding'] = None,
    ) -> typing.Tuple[
        BaseModel,
        BasePreprocessor,
        DatasetBuilder,
        DataLoaderBuilder,
    ]:
        """
        Prepare.

        :param model_class: Model class.
        :param data_pack: DataPack used to fit the preprocessor.
        :param callback: Callback used to padding a batch.
            (default: the default callback of `model_class`)
        :param preprocessor: Preprocessor used to fit the `data_pack`.
            (default: the default preprocessor of `model_class`)

        :return: A tuple of `(model, preprocessor, dataset_builder,
            dataloader_builder)`.

        """
        if not callback:
            callback = model_class.get_default_padding_callback()
        if not preprocessor:
            preprocessor = model_class.get_default_preprocessor()

        preprocessor.fit(data_pack, verbose=0)

        model, embedding_matrix = self._build_model(
            model_class,
            preprocessor,
            embedding
        )

        dataset_builder = self._build_dataset_builder(
            model,
            embedding_matrix,
            preprocessor
        )

        dataloader_builder = self._build_dataloader_builder(
            model,
            callback
        )

        return (
            model,
            preprocessor,
            dataset_builder,
            dataloader_builder
        )

    def _build_model(
        self,
        model_class,
        preprocessor,
        embedding
    ) -> typing.Tuple[BaseModel, np.ndarray]:

        model = model_class()
        model.params['task'] = self._task

        if 'with_embedding' in model.params:
            embedding_matrix = self._build_matrix(preprocessor, embedding)
            model.params['embedding'] = embedding_matrix
        else:
            embedding_matrix = None

        model.build()

        return model, embedding_matrix

    def _build_matrix(self, preprocessor, embedding):
        if embedding is not None:
            vocab_unit = preprocessor.context['vocab_unit']
            term_index = vocab_unit.state['term_index']
            return embedding.build_matrix(term_index)
        else:
            matrix_shape = (
                preprocessor.context['vocab_size'],
                self._config['embedding_output_dim']
            )
            return np.random.uniform(-0.2, 0.2, matrix_shape)

    def _build_dataset_builder(self, model, embedding_matrix, preprocessor):
        builder_kwargs = dict(
            callbacks=[],
            batch_size=self._config['batch_size'],
            shuffle=self._config['shuffle'],
            sort=self._config['sort']
        )

        if isinstance(self._task.losses[0], (mz.losses.RankHingeLoss,
                                             mz.losses.RankCrossEntropyLoss)):
            builder_kwargs.update(dict(
                mode='pair',
                num_dup=self._config['num_dup'],
                num_neg=self._config['num_neg'],
                resample=self._config['resample'],
            ))

        if isinstance(model, mz.models.CDSSM):
            triletter_callback = mz.dataloader.callbacks.Ngram(
                preprocessor, mode='sum')
            builder_kwargs['callbacks'].append(triletter_callback)

        if isinstance(model, mz.models.DSSM):
            triletter_callback = mz.dataloader.callbacks.Ngram(
                preprocessor, mode='aggregate')
            builder_kwargs['callbacks'].append(triletter_callback)

        if isinstance(model, mz.models.DUET):
            triletter_callback = mz.dataloader.callbacks.Ngram(
                preprocessor, mode='sum')
            builder_kwargs['callbacks'].append(triletter_callback)

        if isinstance(model, mz.models.DIIN):
            letter_callback = mz.dataloader.callbacks.Ngram(
                preprocessor, mode='index')
            builder_kwargs['callbacks'].append(letter_callback)

        if isinstance(model, mz.models.DRMM):
            histo_callback = mz.dataloader.callbacks.Histogram(
                embedding_matrix=embedding_matrix,
                bin_size=self._config['bin_size'],
                hist_mode=self._config['hist_mode']
            )
            builder_kwargs['callbacks'].append(histo_callback)

        return DatasetBuilder(**builder_kwargs)

    def _build_dataloader_builder(self, model, callback):
        builder_kwargs = dict(
            stage=self._config['stage'],
            callback=callback
        )
        return DataLoaderBuilder(**builder_kwargs)

    def _infer_num_neg(self):
        if isinstance(self._task.losses[0], (mz.losses.RankHingeLoss,
                                             mz.losses.RankCrossEntropyLoss)):
            self._config['num_neg'] = self._task.losses[0].num_neg

    @classmethod
    def get_default_config(cls) -> dict:
        """Default config getter."""
        return {
            # pair dataset builder kwargs
            'num_dup': 1,

            # dataloader builder kwargs
            'batch_size': 8,
            'stage': 'train',
            'resample': True,
            'shuffle': False,
            'sort': True,

            # histogram unit of DRMM
            'bin_size': 30,
            'hist_mode': 'LCH',

            # dynamic Pooling of MatchPyramid
            'compress_ratio_left': 1.0,
            'compress_ratio_right': 1.0,

            # if no `matchzoo.Embedding` is passed to `tune`
            'embedding_output_dim': 100
        }
