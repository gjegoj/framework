"""Per-stage sources: a partition decided upstream is used as given, never re-cut."""

from __future__ import annotations

import pandas as pd
import pytest

from src.core import Stage
from src.data import DeclaredSource, TableDataModule, TableSource, random_split
from tests.support.tables import label_schema, labelled


def stage_sources() -> list[DeclaredSource]:
    return [
        DeclaredSource(labelled(["cat", "dog", "cat", "dog"]), stage=Stage.TRAIN),
        DeclaredSource(labelled(["cat", "dog"]), stage=Stage.VAL),
        DeclaredSource(labelled(["cat"]), stage=Stage.TEST),
    ]


def test_each_stage_keeps_exactly_the_rows_its_own_source_declared() -> None:
    module = TableDataModule(sources=stage_sources(), schema=label_schema())

    module.setup()

    assert [len(module.dataset(stage)) for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)] == [4, 2, 1]


def test_encoders_still_fit_on_train_only() -> None:
    """The leakage guard is a property of the module, not of the way stages were obtained."""
    module = TableDataModule(
        sources=[
            DeclaredSource(labelled(["cat", "dog"]), stage=Stage.TRAIN),
            DeclaredSource(labelled(["dog", "cat"]), stage=Stage.VAL),
        ],
        schema=label_schema(),
    )
    facts = module.setup()

    assert facts["label"].num_classes == 2
    assert facts["label"].class_names == ("cat", "dog")


def test_a_val_value_outside_the_declared_classes_is_refused_at_setup_naming_the_stage() -> None:
    """Encoders fit on train; the other splits are validated against what was declared, so a
    value only val carries dies here rather than in the first validation epoch."""
    module = TableDataModule(
        sources=[
            DeclaredSource(labelled(["cat", "dog"]), stage=Stage.TRAIN),
            DeclaredSource(labelled(["cat", "dog", "unseen_in_train"]), stage=Stage.VAL),
        ],
        schema=label_schema(),
    )

    with pytest.raises(LookupError, match=r"val.*unseen_in_train"):
        module.setup()


def test_a_splitter_alongside_per_stage_sources_is_refused() -> None:
    with pytest.raises(ValueError, match="nothing to divide"):
        TableDataModule(
            sources=stage_sources(),
            schema=label_schema(),
            splitter=random_split({Stage.TRAIN: 1.0}, seed=42),
        )


def test_a_single_source_without_a_splitter_is_refused() -> None:
    with pytest.raises(ValueError, match="divided into stages"):
        TableDataModule(sources=[DeclaredSource(labelled(["cat", "dog"]))], schema=label_schema())


def test_stages_without_train_are_refused_because_encoders_need_it() -> None:
    module = TableDataModule(sources=[DeclaredSource(labelled(["cat"]), stage=Stage.VAL)], schema=label_schema())

    with pytest.raises(ValueError, match="No train rows"):
        module.setup()


def test_sources_are_read_at_setup_not_at_construction() -> None:
    """Construction is eager; reading waits until the run is seeded."""

    class ExplodingSource(TableSource):
        def read(self) -> pd.DataFrame:
            raise AssertionError("read() must not run during construction")

    TableDataModule(sources=[DeclaredSource(ExplodingSource(), stage=Stage.TRAIN)], schema=label_schema())


def test_a_pinned_stage_may_sit_beside_the_sources_the_splitter_divides() -> None:
    """A held-out test set beside a pool the run divides into train and val — a partition
    decided upstream for one stage only, which neither layout alone could express."""
    module = TableDataModule(
        sources=[
            DeclaredSource(labelled(["cat", "dog"] * 4)),
            DeclaredSource(labelled(["cat"]), stage=Stage.TEST),
        ],
        schema=label_schema(),
        splitter=random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.5}, seed=42),
    )
    module.setup()

    assert [len(module.dataset(stage)) for stage in (Stage.TRAIN, Stage.VAL, Stage.TEST)] == [4, 4, 1]
