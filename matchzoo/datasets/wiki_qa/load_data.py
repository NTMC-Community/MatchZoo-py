"""WikiQA data loader."""

import typing
from pathlib import Path

import os
import csv
import wget
import zipfile
import pandas as pd

import matchzoo

_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(
    stage: str = 'train',
    task: str = 'ranking',
    filtered: bool = False,
    return_classes: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    file_path = _download_data(stage)
    data_root = file_path.parent
    data_pack = _read_data(file_path)
    if filtered and stage in ('dev', 'test'):
        ref_path = data_root.joinpath(f'WikiQA-{stage}.ref')
        filter_ref_path = data_root.joinpath(f'WikiQA-{stage}-filtered.ref')
        with open(filter_ref_path, mode='r') as f:
            filtered_ids = set([line.split()[0] for line in f])
        filtered_lines = []
        with open(ref_path, mode='r') as f:
            for idx, line in enumerate(f.readlines()):
                if line.split()[0] in filtered_ids:
                    filtered_lines.append(idx)
        data_pack = data_pack[filtered_lines]

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data(stage):
    target_dir = matchzoo.USER_DATA_DIR.joinpath('wiki_qa')
    zip_path = target_dir.joinpath('wikiqa.zip')
    target = target_dir.joinpath(
        'WikiQACorpus').joinpath(f'WikiQA-{stage}.tsv')

    if os.path.exists(target):
        return target
    elif os.path.exists(zip_path):
        return unzip(zip_path, stage)
    else:
        return unzip(download(_url, target_dir), stage)


def unzip(zip_path, stage):
    dirpath = zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            zf.extract(name, dirpath)
    return dirpath.joinpath('WikiQACorpus').joinpath(f'WikiQA-{stage}.tsv')


def download(url, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    zip_path = os.path.join(target_dir, 'wikiqa.zip')
    wget.download(url, zip_path)
    return Path(zip_path)


def _read_data(path):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['Question'],
        'text_right': table['Sentence'],
        'id_left': table['QuestionID'],
        'id_right': table['SentenceID'],
        'label': table['Label']
    })
    return matchzoo.pack(df)
