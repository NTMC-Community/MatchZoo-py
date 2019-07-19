import torch
import pytest
from pathlib import Path
import shutil

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
def save_dir():
    return Path('.matchzoo_test_save_load_tmpdir')


@pytest.fixture(scope='module')
def trainer(
    model, optimizer, dataloader, scheduler, save_dir
):
    return mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=dataloader,
        validloader=dataloader,
        epochs=4,
        validate_interval=2,
        patience=1,
        scheduler=scheduler,
        clip_norm=10,
        save_dir=save_dir,
        save_all=True,
        verbose=1,
    )


@pytest.mark.slow
def test_trainer(trainer, dataloader, save_dir):
    trainer.run()
    assert trainer.evaluate(dataloader)
    assert trainer.predict(dataloader) is not None

    # Save model
    model_checkpoint = save_dir.joinpath('model.pt')
    trainer.save_model()
    trainer.restore_model(model_checkpoint)

    # Save model
    trainer_checkpoint = save_dir.joinpath('trainer.pt')
    trainer.save()
    trainer.restore(trainer_checkpoint)

    if save_dir.exists():
        shutil.rmtree(save_dir)
