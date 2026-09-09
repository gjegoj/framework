"""Combining datasets that need different handling, each keeping its own transforms."""

from __future__ import annotations

import logging
from collections.abc import Sized
from typing import cast

import pytest

from src.core import Sample, Stage
from src.core.ports import SampleTransform
from src.data import DeclaredSource, TableDataModule, random_split
from tests.support.tables import label_schema, repeated

FRACTIONS = {Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}


def marker(value: float) -> SampleTransform:
    def transform(sample: Sample) -> Sample:
        sample.inputs["image"] = sample.inputs["image"] + value
        return sample

    return transform


def size_of(module: TableDataModule, stage: Stage) -> int:
    return len(cast("Sized", module.dataset(stage)))


def markers_in(module: TableDataModule, stage: Stage) -> set[float]:
    """The first pixel of every sample in a stage — the marker each transform writes.

    Which transform ran is read off the pixels rather than asserted on the pipeline
    object, so the test proves the rows went through it.
    """
    dataset = module.dataset(stage)
    return {float(dataset[index].inputs["image"][0]) for index in range(size_of(module, stage))}


def test_each_source_keeps_the_transform_it_declared() -> None:
    """The whole point: rows from a noisier set can be handled differently."""
    module = TableDataModule(
        sources=[
            DeclaredSource(repeated(4, "cat"), transforms={Stage.TRAIN: marker(1.0)}),
            DeclaredSource(repeated(4, "dog"), transforms={Stage.TRAIN: marker(9.0)}),
        ],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )
    module.setup()

    seen = markers_in(module, Stage.TRAIN)

    assert seen == {1.0, 9.0}


def test_a_source_without_its_own_transform_takes_the_stage_one() -> None:
    module = TableDataModule(
        sources=[
            DeclaredSource(repeated(4, "cat"), transforms={Stage.TRAIN: marker(1.0)}),
            DeclaredSource(repeated(4, "dog")),
        ],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
        transforms={Stage.TRAIN: marker(5.0)},
    )
    module.setup()

    seen = markers_in(module, Stage.TRAIN)

    assert seen == {1.0, 5.0}


def test_every_source_reaches_every_stage() -> None:
    """Sources are divided one by one, so a small one cannot land wholly in train."""
    module = TableDataModule(
        sources=[DeclaredSource(repeated(100, "cat")), DeclaredSource(repeated(8, "dog"))],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )
    module.setup()

    test = module.dataset(Stage.TEST)
    labels = {test[index].targets["label"] for index in range(size_of(module, Stage.TEST))}

    assert len(labels) == 2


def test_encoders_fit_on_the_train_rows_of_every_source() -> None:
    """A vocabulary that spanned only the first source would reject the rest at encode time."""
    module = TableDataModule(
        sources=[DeclaredSource(repeated(8, "cat")), DeclaredSource(repeated(8, "dog"))],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )
    facts = module.setup()

    assert facts["label"].class_names == ("cat", "dog")


def test_the_stages_add_up_to_every_row_of_every_source() -> None:
    module = TableDataModule(
        sources=[DeclaredSource(repeated(8)), DeclaredSource(repeated(4))],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )
    module.setup()

    assert sum(size_of(module, stage) for stage in Stage) == 12


def test_a_single_source_stays_a_single_dataset() -> None:
    """The common case pays nothing for a feature it does not use."""
    from src.data.dataset import TableDataset

    module = TableDataModule(
        sources=[DeclaredSource(repeated(8))],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )
    module.setup()

    assert isinstance(module.dataset(Stage.TRAIN), TableDataset)


def test_a_source_transform_is_announced(caplog: pytest.LogCaptureFixture) -> None:
    """It replaces the stage pipeline, so a run should say where that happens."""
    module = TableDataModule(
        sources=[DeclaredSource(repeated(8), transforms={Stage.TRAIN: marker(1.0)})],
        schema=label_schema(),
        splitter=random_split(FRACTIONS, seed=42),
    )

    with caplog.at_level(logging.INFO):
        module.setup()

    assert "own transform" in caplog.text


def test_per_stage_sources_may_carry_transforms_too() -> None:
    module = TableDataModule(
        sources=[
            DeclaredSource(repeated(4, "cat"), stage=Stage.TRAIN, transforms={Stage.TRAIN: marker(1.0)}),
            DeclaredSource(repeated(4, "dog"), stage=Stage.TRAIN),
            DeclaredSource(repeated(2, "cat"), stage=Stage.VAL),
        ],
        schema=label_schema(),
        transforms={Stage.TRAIN: marker(5.0)},
    )
    module.setup()

    seen = markers_in(module, Stage.TRAIN)

    assert seen == {1.0, 5.0}
