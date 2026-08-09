"""``TrainingData``: per-stage DataLoaders from an already-set-up ``DataModule``."""

from __future__ import annotations

import logging
from typing import Any, override

import pandas as pd
import pytest
import torch
from torch import Tensor
from torch.utils.data import RandomSampler, SequentialSampler

from src.core import Batch, DataProfile, Stage
from src.core.ports import DataModule
from src.data import (
    DataSchema,
    InMemorySource,
    InputColumn,
    LabelTargetEncoder,
    TableDataModule,
    TargetColumn,
    random_split,
)
from src.training import TrainingData


def load_point(value: Any) -> Tensor:
    return torch.tensor([float(value), 1.0])


def make_data_module() -> TableDataModule:
    table = pd.DataFrame({"x": [float(index) for index in range(8)], "label": ["cat", "dog"] * 4})
    module = TableDataModule(
        source=InMemorySource(table),
        schema=DataSchema(
            inputs={"point": InputColumn(column="x", loader=load_point)},
            targets={"label": TargetColumn(column="label", encoder=LabelTargetEncoder())},
        ),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42),
    )
    module.setup(DataProfile())
    return module


class TrainAndValOnly(DataModule):
    """A pipeline that honestly carries no test split, as a YOLO descriptor often does."""

    def __init__(self, complete: TableDataModule) -> None:
        self._complete = complete

    @override
    def setup(self, profile: DataProfile) -> None: ...

    @override
    def dataset(self, stage: Stage) -> Any:
        if stage is Stage.TEST:
            raise LookupError("No dataset for stage 'test'. Available stages: train, val.")
        return self._complete.dataset(stage)


def test_a_run_with_no_test_data_is_tested_on_its_validation_set() -> None:
    """Falling back lets the run finish and report, instead of dying with the weights trained.

    A pipeline may honestly carry no test split: a YOLO descriptor often ships
    without one, and per-stage sources need not declare all three.
    """
    partial = TrainAndValOnly(make_data_module())
    data = TrainingData(partial, batch_size=2)

    assert data.test_dataloader().dataset is partial.dataset(Stage.VAL)


def test_the_substitution_of_val_for_test_is_said_out_loud(caplog: pytest.LogCaptureFixture) -> None:
    """Silence here publishes an optimistic number under an honest name.

    Every `test/*` scalar would be computed on the rows the checkpoint was selected
    on, reported under the one stage name that is supposed to mean held-out data.
    """
    data = TrainingData(TrainAndValOnly(make_data_module()), batch_size=2)

    with caplog.at_level(logging.WARNING):
        data.test_dataloader()

    said = [record.message for record in caplog.records if "no test data" in record.message]
    assert len(said) == 1
    assert "validation set" in said[0]


def test_a_run_that_has_test_data_is_tested_on_it_and_says_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback must not fire — nor warn — when there is nothing to fall back from."""
    complete = make_data_module()
    data = TrainingData(complete, batch_size=2)

    with caplog.at_level(logging.WARNING):
        loader = data.test_dataloader()

    assert loader.dataset is complete.dataset(Stage.TEST)
    assert not [record for record in caplog.records if "no test data" in record.message]


def test_train_loader_shuffles_and_collates_batches() -> None:
    data = TrainingData(make_data_module(), batch_size=2)

    loader = data.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(loader.sampler, RandomSampler)
    assert isinstance(batch, Batch)
    assert batch.inputs["point"].shape == (2, 2)


def test_evaluation_loaders_keep_the_order() -> None:
    data = TrainingData(make_data_module(), batch_size=2)

    assert isinstance(data.val_dataloader().sampler, SequentialSampler)
    assert isinstance(data.test_dataloader().sampler, SequentialSampler)


def test_unknown_loader_options_reach_torch() -> None:
    """The forwarding convention: a knob the adapter never declared still works."""
    data = TrainingData(make_data_module(), batch_size=2, num_workers=1, prefetch_factor=2)

    assert data.train_dataloader().prefetch_factor == 2


def test_only_training_drops_the_last_batch() -> None:
    """A truncated val/test split would mean metrics computed on part of the data."""
    data = TrainingData(make_data_module(), batch_size=3, drop_last=True)

    assert data.train_dataloader().drop_last is True
    assert data.val_dataloader().drop_last is False
    assert data.test_dataloader().drop_last is False
