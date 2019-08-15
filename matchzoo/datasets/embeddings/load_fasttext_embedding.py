"""FastText embedding data loader."""

from pathlib import Path

import matchzoo as mz

_fasttext_embedding_url = "https://dl.fbaipublicfiles.com/fasttext/vectors" \
                          "-wiki/wiki.{}.vec"


def load_fasttext_embedding(language: str = 'en') -> mz.embedding.Embedding:
    """
    Return the pretrained fasttext embedding.

    :param language: the language of embedding. Supported language can be
        referred to "https://github.com/facebookresearch/fastText/blob/master"
        "/docs/pretrained-vectors.md"
    :return: The :class:`mz.embedding.Embedding` object.
    """
    file_name = _fasttext_embedding_url.split('/')[-1].format(language)
    file_path = (Path(mz.USER_DATA_DIR) / 'fasttext').joinpath(file_name)
    if not file_path.exists():
        mz.utils.get_file(file_name,
                          _fasttext_embedding_url.format(language),
                          extract=False,
                          cache_dir=mz.USER_DATA_DIR,
                          cache_subdir='fasttext')
    return mz.embedding.load_from_file(file_path=str(file_path),
                                       mode='fasttext')
