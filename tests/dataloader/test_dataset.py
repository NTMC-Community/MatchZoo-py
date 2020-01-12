import math

import numpy as np

import matchzoo as mz


def test_dataset():
    data_pack = mz.datasets.toy.load_data('train', task='ranking')
    preprocessor = mz.preprocessors.BasicPreprocessor()
    data_processed = preprocessor.fit_transform(data_pack)

    dataset_point = mz.dataloader.Dataset(
        data_processed,
        mode='point',
        batch_size=1,
        resample=False,
        shuffle=True,
        sort=False
    )
    dataset_point.batch_size = 10
    dataset_point.shuffle = not dataset_point.shuffle
    dataset_point.sort = not dataset_point.sort
    assert len(dataset_point.batch_indices) == 10

    dataset_pair = mz.dataloader.Dataset(
        data_processed,
        mode='pair',
        num_dup=1,
        num_neg=1,
        batch_size=1,
        resample=True,
        shuffle=False,
        sort=False
    )
    assert len(dataset_pair) == 5
    dataset_pair.num_dup = dataset_pair.num_dup + 1
    assert len(dataset_pair) == 10
    dataset_pair.num_neg = dataset_pair.num_neg + 2
    assert len(dataset_pair) == 10
    dataset_pair.batch_size = dataset_pair.batch_size + 1
    assert len(dataset_pair) == 5
    dataset_pair.resample = not dataset_pair.resample
    assert len(dataset_pair) == 5


def test_dataloader_length():
    data_pack = mz.datasets.toy.load_data('train', task='ranking')
    preprocessor = mz.preprocessors.BasicPreprocessor()
    data_processed = preprocessor.fit_transform(data_pack)
    batch_size = 1
    dataset_point = mz.dataloader.Dataset(
        data_processed,
        mode='point',
        batch_size=batch_size,
        resample=False,
        shuffle=True,
        sort=False
    )

    num_batches = math.ceil(len(data_pack.relation) / batch_size)
    assert len(dataset_point) == num_batches

    dataloader = mz.dataloader.DataLoader(dataset_point)
    loaded_data = [d for d in dataloader]
    assert len(loaded_data) == len(dataloader) == num_batches


    dataloader = mz.dataloader.DataLoader(dataset_point, num_workers=2)
    loaded_data = [d for d in dataloader]
    assert len(loaded_data) == len(dataloader) == num_batches


def test_shuffle():
    data_pack = mz.datasets.toy.load_data('train', task='ranking')
    preprocessor = mz.preprocessors.BasicPreprocessor()
    data_processed = preprocessor.fit_transform(data_pack)
    batch_size = 1

    # check shuffled
    dataset_point = mz.dataloader.Dataset(
        data_processed,
        mode='point',
        batch_size=batch_size,
        resample=False,
        shuffle=True,
        sort=False
    )
    dataloader = mz.dataloader.DataLoader(dataset_point)
    loaded_labels = [label.item() for x, label in dataloader]
    loaded_labels2 = [label.item() for x, label in dataloader]
    assert not np.array_equal(loaded_labels, loaded_labels2)

    # check not shuffled
    dataset_point = mz.dataloader.Dataset(
        data_processed,
        mode='point',
        batch_size=batch_size,
        resample=False,
        shuffle=False,
        sort=False
    )
    dataloader = mz.dataloader.DataLoader(dataset_point)
    loaded_labels = [label.item() for x, label in dataloader]
    loaded_labels2 = [label.item() for x, label in dataloader]
    assert np.array_equal(loaded_labels, loaded_labels2)