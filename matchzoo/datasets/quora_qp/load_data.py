"""Quora Question Pairs data loader."""

import typing
from pathlib import Path

import os
import wget
import zipfile
import pandas as pd

import matchzoo

_url = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence" \
       "-representations.appspot.com/o/data%2FQQP.zip?alt=media&" \
       "token=700c6acf-160d-4d89-81d1-de4191d02cb5"


def load_data(
    stage: str = 'train',
    task: str = 'classification',
    return_classes: bool = False,
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load QuoraQP data.

    :param path: `None` for _download from quora, specific path for
        downloaded data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param return_classes: Whether return classes for classification task.
    :return: A DataPack if `ranking`, a tuple of (DataPack, classes) if
        `classification`.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    file_path = _download_data(stage)
    data_pack = _read_data(file_path, stage)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    elif task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task.")


def _download_data(stage):
    target_dir = matchzoo.USER_DATA_DIR.joinpath('quora_qp')
    zip_path = target_dir.joinpath('QQP.zip')
    target = target_dir.joinpath('QQP').joinpath(f'{stage}.tsv')

    if os.path.exists(target):
        return target
    elif os.path.exists(zip_path):
        return _unzip(zip_path, stage)
    else:
        return _unzip(_download(_url, target_dir), stage)


def _unzip(zip_path, stage):
    dirpath = zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            zf.extract(name, dirpath)
    return dirpath.joinpath('QQP').joinpath(f'{stage}.tsv')


def _download(url, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    zip_path = os.path.join(target_dir, 'QQP.zip')
    wget.download(url, zip_path)
    return Path(zip_path)


def _read_data(path, stage):
    data = pd.read_csv(path, sep='\t', error_bad_lines=False, dtype=object)
    data = data.dropna(axis=0, how='any').reset_index(drop=True)
    if stage in ['train', 'dev']:
        df = pd.DataFrame({
            'id_left': data['qid1'],
            'id_right': data['qid2'],
            'text_left': data['question1'],
            'text_right': data['question2'],
            'label': data['is_duplicate'].astype(int)
        })
    else:
        df = pd.DataFrame({
            'text_left': data['question1'],
            'text_right': data['question2']
        })
    return matchzoo.pack(df)
