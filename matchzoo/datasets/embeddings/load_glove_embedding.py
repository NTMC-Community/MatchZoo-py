"""Embedding data loader."""

from pathlib import Path

import os
import wget
import zipfile

import matchzoo as mz

_glove_embedding_url = "http://nlp.stanford.edu/data/glove.6B.zip"


def load_glove_embedding(dimension: int = 50) -> mz.embedding.Embedding:
    """
    Return the pretrained glove embedding.

    :param dimension: the size of embedding dimension, the value can only be
        50, 100, or 300.
    :return: The :class:`mz.embedding.Embedding` object.
    """
    glove_dir = mz.USER_DATA_DIR.joinpath('glove')
    file_name = 'glove.6B.' + str(dimension) + 'd.txt'
    file_path = glove_dir.joinpath(file_name)

    if not os.path.exists(file_path):
        zip_path = glove_dir.joinpath('glove.6B.zip')
        if os.path.exists(zip_path):
            unzip(zip_path)
        else:
            unzip(download(_glove_embedding_url, glove_dir))
    return mz.embedding.load_from_file(file_path=str(file_path), mode='glove')


def unzip(filepath):
    dirpath = filepath.parent
    with zipfile.ZipFile(filepath) as zf:
        for name in zf.namelist():
            zf.extract(name, dirpath)


def download(url, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    filepath = os.path.join(target_dir, 'glove.6B.zip')
    wget.download(url, filepath)
    return Path(filepath)
