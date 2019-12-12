import pytest

import matchzoo as mz
from matchzoo import preprocessors
from matchzoo.dataloader import callbacks
from matchzoo.dataloader import Dataset, DataLoader
from matchzoo.datasets import embeddings
from matchzoo.embedding import load_from_file


@pytest.fixture(scope='module')
def train_raw():
    return mz.datasets.toy.load_data('test', task='ranking')[:5]


def test_basic_padding(train_raw):
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(train_raw, verbose=0)
    dataset = Dataset(data_preprocessed, batch_size=5, mode='point')

    pre_fixed_padding = callbacks.BasicPadding(
        fixed_length_left=5, fixed_length_right=5, pad_word_mode='pre', with_ngram=False)
    dataloader = DataLoader(dataset, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)

    post_padding = callbacks.BasicPadding(pad_word_mode='post', with_ngram=False)
    dataloader = DataLoader(dataset, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].detach().cpu().numpy())
        max_right_len = max(batch[0]['length_right'].detach().cpu().numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)


def test_drmm_padding(train_raw):
    preprocessor = preprocessors.BasicPreprocessor()
    data_preprocessed = preprocessor.fit_transform(train_raw, verbose=0)

    embedding_matrix = load_from_file(embeddings.EMBED_10_GLOVE, mode='glove')
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = embedding_matrix.build_matrix(term_index)
    histgram_callback = callbacks.Histogram(
        embedding_matrix=embedding_matrix, bin_size=30, hist_mode='LCH')
    dataset = Dataset(
        data_preprocessed, mode='point', batch_size=5, callbacks=[histgram_callback])

    pre_fixed_padding = callbacks.DRMMPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(dataset, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 5)
        assert batch[0]['text_right'].shape == (5, 5)
        assert batch[0]['match_histogram'].shape == (5, 5, 30)

    post_padding = callbacks.DRMMPadding(pad_mode='post')
    dataloader = DataLoader(dataset, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].detach().cpu().numpy())
        max_right_len = max(batch[0]['length_right'].detach().cpu().numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len)
        assert batch[0]['text_right'].shape == (5, max_right_len)
        assert batch[0]['match_histogram'].shape == (5, max_left_len, 30)


def test_bert_padding(train_raw):
    preprocessor = preprocessors.BertPreprocessor()
    data_preprocessed = preprocessor.transform(train_raw, verbose=0)
    dataset = Dataset(data_preprocessed, mode='point', batch_size=5)

    pre_fixed_padding = callbacks.BertPadding(
        fixed_length_left=5, fixed_length_right=5, pad_mode='pre')
    dataloader = DataLoader(dataset, callback=pre_fixed_padding)
    for batch in dataloader:
        assert batch[0]['text_left'].shape == (5, 7)
        assert batch[0]['text_right'].shape == (5, 6)

    post_padding = callbacks.BertPadding(pad_mode='post')
    dataloader = DataLoader(dataset, callback=post_padding)
    for batch in dataloader:
        max_left_len = max(batch[0]['length_left'].detach().cpu().numpy())
        max_right_len = max(batch[0]['length_right'].detach().cpu().numpy())
        assert batch[0]['text_left'].shape == (5, max_left_len + 2)
        assert batch[0]['text_right'].shape == (5, max_right_len + 1)
