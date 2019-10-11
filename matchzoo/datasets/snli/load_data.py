"""SNLI data loader."""

import typing
from pathlib import Path

import pandas as pd

import matchzoo
from matchzoo.engine.base_task import BaseTask

_url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"


def load_data(
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'classification',
    target_label: str = 'entailment',
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load SNLI data.

    :param stage: One of `train`, `dev`, and `test`. (default: `train`)
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance. (default: `classification`)
    :param target_label: If `ranking`, chose one of `entailment`,
        `contradiction` and `neutral` as the positive label.
        (default: `entailment`)
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f'snli_1.0_{stage}.txt')
    data_pack = _read_data(file_path, task, target_label)

    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        classes = ['entailment', 'contradiction', 'neutral']
        if return_classes:
            return data_pack, classes
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = matchzoo.utils.get_file(
        'snli', _url, extract=True,
        cache_dir=matchzoo.USER_DATA_DIR,
        cache_subdir='snli'
    )
    return Path(ref_path).parent.joinpath('snli_1.0')


def _read_data(path, task, target_label):
    table = pd.read_csv(path, sep='\t')
    df = pd.DataFrame({
        'text_left': table['sentence1'],
        'text_right': table['sentence2'],
        'label': table['gold_label']
    })
    df = df.dropna(axis=0, how='any').reset_index(drop=True)

    filter_id = df[df['label'] == '-'].index.tolist()
    df.drop(filter_id, inplace=True)

    if task == 'ranking' or isinstance(task, matchzoo.tasks.Ranking):
        if target_label not in ['entailment', 'contradiction', 'neutral']:
            raise ValueError(f"{target_label} is not a valid target label."
                             f"Must be one of `entailment`, `contradiction`"
                             f" and `neutral`")
        df['label'] = (df['label'] == target_label)
    elif task == 'classification' or isinstance(
            task, matchzoo.tasks.Classification):
        classes = ['entailment', 'contradiction', 'neutral']
        df['label'] = df['label'].apply(classes.index)
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")

    return matchzoo.pack(df, task)
