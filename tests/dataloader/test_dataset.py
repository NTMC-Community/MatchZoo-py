import matchzoo as mz
from matchzoo import preprocessors
from matchzoo.dataloader import Dataset


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
