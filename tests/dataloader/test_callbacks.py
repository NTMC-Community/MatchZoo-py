import pytest

import matchzoo as mz
from matchzoo import preprocessors
from matchzoo.dataloader import callbacks
from matchzoo.dataloader import Dataset, DataLoader
from matchzoo.datasets import embeddings
from matchzoo.embedding import load_from_file


def test_basic_padding():
    data = mz.datasets.toy.load_data('test', task='ranking')[:5]
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(data, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point')

    pre_fixed_padding = callbacks.BasicPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)

    post_padding = callbacks.BasicPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)


def test_drmm_padding():
    data = mz.datasets.toy.load_data('test', task='ranking')[:5]
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(data, verbose=0)

    embedding_matrix = load_from_file(embeddings.EMBED_10_GLOVE, mode='glove')
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = embedding_matrix.build_matrix(term_index)
    histgram_callback = callbacks.Histogram(
        embedding_matrix=embedding_matrix, bin_size=30, hist_mode='LCH')
    dataset = Dataset(
        data_preprocessed, mode='point', callbacks=[histgram_callback])

    pre_fixed_padding = callbacks.DRMMPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)
        assert batch[0]['match_histogram'].shape == (5, 5, 30)

    post_padding = callbacks.DRMMPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)
        assert batch[0]['match_histogram'].shape == (5, max_left_len, 30)


def test_cdssm_padding():
    data = mz.datasets.toy.load_data('test', task='ranking')[:5]
    preprocessor = preprocessors.CDSSMPreprocessor()
    data_preprocessed = preprocessor.fit_transform(data, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point')

    pre_fixed_padding = callbacks.CDSSMPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        vocab_size = preprocessor.context['vocab_size']
        assert batch[0]['text_left'].shape == (5, 5, vocab_size)
        assert batch[0]['text_right'].shape == (5, 5, vocab_size)

    post_padding = callbacks.CDSSMPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        vocab_size = preprocessor.context['vocab_size']
        assert batch[0]['text_left'].shape == (5, max_left_len, vocab_size)
        assert batch[0]['text_right'].shape == (5, max_right_len, vocab_size)


def test_bert_padding():
    data = mz.datasets.toy.load_data('test', task='ranking')[:5]
    preprocessor = preprocessors.BertPreprocessor()
    data_preprocessed = preprocessor.transform(data, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point')

    pre_fixed_padding = callbacks.BertPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(
        dataset, batch_size=5, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 6)
        assert batch[0]['text_right'].shape == (5, 7)

    post_padding = callbacks.BertPadding(pad_mode='post')
    dataloader = DataLoader(dataset, batch_size=5, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].numpy())
        max_right_len = max(batch[0]['length_right'].numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len + 1)
        assert batch[0]['text_right'].shape == (5, max_right_len + 2)
