"""Per-stage sources: a partition decided upstream is used as given, never re-cut."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core import DataProfile, Stage
from src.data import (
    InMemorySource,
    TableDataModule,
    random_split,
)
from tests.support.tables import label_schema, labelled


def stage_sources() -> dict[Stage, InMemorySource]:
    return {
        Stage.TRAIN: InMemorySource(labelled(["cat", "dog", "cat", "dog"])),
        Stage.VAL: InMemorySource(labelled(["cat", "dog"])),
        Stage.TEST: InMemorySource(labelled(["cat"])),
    }


def test_each_stage_keeps_exactly_the_rows_its_own_source_declared() -> None:
    module = TableDataModule(source=stage_sources(), schema=label_schema())

    module.setup(DataProfile())

    assert [len(module.dataset(stage)) for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)] == [4, 2, 1]


def test_encoders_still_fit_on_train_only() -> None:
    """The leakage guard is a property of the module, not of the way stages were obtained."""
    module = TableDataModule(
        source={
            Stage.TRAIN: InMemorySource(labelled(["cat", "dog"])),
            Stage.VAL: InMemorySource(labelled(["cat", "dog", "unseen_in_train"])),
        },
        schema=label_schema(),
    )
    profile = DataProfile()

    module.setup(profile)

    assert profile.facts("label").num_classes == 2
    assert profile.facts("label").class_names == ["cat", "dog"]


def test_a_splitter_alongside_per_stage_sources_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to divide"):
        TableDataModule(
            source=stage_sources(),
            schema=label_schema(),
            splitter=random_split({Stage.TRAIN: 1.0}, seed=42),
        )


def test_a_single_source_without_a_splitter_is_refused() -> None:
    with pytest.raises(ValueError, match="divided into stages"):
        TableDataModule(source=InMemorySource(labelled(["cat", "dog"])), schema=label_schema())


def test_stages_without_train_are_refused_because_encoders_need_it() -> None:
    module = TableDataModule(source={Stage.VAL: InMemorySource(labelled(["cat"]))}, schema=label_schema())

    with pytest.raises(ValueError, match="No train rows"):
        module.setup(DataProfile())


def test_sources_are_read_at_setup_not_at_construction() -> None:
    """Assembly builds; reading waits until the run is seeded."""

    class ExplodingSource(InMemorySource):
        def read(self) -> pd.DataFrame:
            raise AssertionError("read() must not run during construction")

    TableDataModule(source={Stage.TRAIN: ExplodingSource(labelled(["cat"]))}, schema=label_schema())
