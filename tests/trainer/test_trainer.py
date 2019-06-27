import torch
import pytest

import matchzoo as mz


@pytest.fixture(scope='module')
def task():
    return mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss())


@pytest.fixture(scope='module')
def train_raw(task):
    return mz.datasets.toy.load_data('train', task)[:10]


@pytest.fixture(scope='module')
def model_class():
    return mz.models.DenseBaseline


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
def scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


@pytest.fixture(scope='module')
def validate_interval():
    return 5


@pytest.fixture(scope='module')
def patience():
    return 1


@pytest.fixture(scope='module')
def clip_norm():
    return 10


@pytest.mark.slow
def test_trainer(
    model, optimizer, dataloader, scheduler,
    validate_interval, patience, clip_norm
):
    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=dataloader,
        validloader=dataloader,
        epochs=3,
        validate_interval=validate_interval,
        patience=patience,
        scheduler=scheduler,
        clip_norm=clip_norm,
        verbose=0
    )
    trainer.run()
    assert trainer.evaluate(dataloader)
    assert trainer.predict(dataloader) is not None
