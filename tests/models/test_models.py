"""
These tests are simplied because the original verion takes too much time to
run, making CI fails as it reaches the time limit.
"""
import torch
import pytest
from pathlib import Path
import shutil

import matchzoo as mz


@pytest.fixture(scope='module', params=[
    mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=2)),
    mz.tasks.Classification(num_classes=2),
])
def task(request):
    return request.param


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)[:10]


@pytest.fixture(scope='module', params=mz.models.list_available())
def model_class(request):
    return request.param


@pytest.fixture(scope='module')
def embedding():
    return mz.datasets.toy.load_embedding()


@pytest.fixture(scope='module')
def setup(task, model_class, train_raw, embedding):
    return mz.auto.prepare(
        task=task,
        model_class=model_class,
        data_pack=train_raw,
        embedding=embedding
    )


@pytest.fixture(scope='module')
def model(setup):
    return setup[0]


@pytest.fixture(scope='module')
def preprocessor(setup):
    return setup[1]


@pytest.fixture(scope='module')
def dataset_builder(setup):
    return setup[2]


@pytest.fixture(scope='module')
def dataloader_builder(setup):
    return setup[3]


@pytest.fixture(scope='module')
def dataloader(train_raw, preprocessor, dataset_builder, dataloader_builder):
    return dataloader_builder.build(
        dataset_builder.build(preprocessor.transform(train_raw)))


@pytest.fixture(scope='module')
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture(scope='module')
def save_dir():
    return Path('.matchzoo_test_save_load_tmpdir')


@pytest.mark.slow
def test_model_fit_eval_predict(model, optimizer, dataloader, save_dir):
    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=dataloader,
        validloader=dataloader,
        epochs=2,
        save_dir=save_dir,
        verbose=0
    )
    trainer.run()

    if save_dir.exists():
        shutil.rmtree(save_dir)
