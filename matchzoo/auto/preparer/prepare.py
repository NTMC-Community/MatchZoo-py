import typing

import matchzoo as mz
from .preparer import Preparer
from matchzoo.engine.base_task import BaseTask
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_callback import BaseCallback
from matchzoo.engine.base_preprocessor import BasePreprocessor


def prepare(
    task: BaseTask,
    model_class: typing.Type[BaseModel],
    data_pack: mz.DataPack,
    callback: typing.Optional[BaseCallback] = None,
    preprocessor: typing.Optional[BasePreprocessor] = None,
    embedding: typing.Optional['mz.Embedding'] = None,
    config: typing.Optional[dict] = None,
):
    """
    A simple shorthand for using :class:`matchzoo.Preparer`.

    `config` is used to control specific behaviors. The default `config`
    will be updated accordingly if a `config` dictionary is passed. e.g. to
    override the default `bin_size`, pass `config={'bin_size': 15}`.

    :param task: Task.
    :param model_class: Model class.
    :param data_pack: DataPack used to fit the preprocessor.
    :param callback: Callback used to padding a batch.
        (default: the default callback of `model_class`)
    :param preprocessor: Preprocessor used to fit the `data_pack`.
        (default: the default preprocessor of `model_class`)
    :param embedding: Embedding to build a embedding matrix. If not set,
        then a correctly shaped randomized matrix will be built.
    :param config: Configuration of specific behaviors. (default: return
        value of `mz.Preparer.get_default_config()`)

    :return: A tuple of `(model, preprocessor, data_generator_builder,
        embedding_matrix)`.

    """
    preparer = Preparer(task=task, config=config)
    return preparer.prepare(
        model_class=model_class,
        data_pack=data_pack,
        callback=callback,
        preprocessor=preprocessor,
        embedding=embedding
    )
