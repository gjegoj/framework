"""Capping rows: ``max_samples`` shrinks a run per source, wherever the source sits."""

from __future__ import annotations

import pytest

from src.core import Sample, Stage
from src.data import DeclaredSource, TableDataModule
from tests.support.tables import label_schema, labelled

ROWS = labelled(["cat", "dog"] * 50)


def capped(max_samples: int | float) -> TableDataModule:  # noqa: PYI041  # a count, or a share
    module = TableDataModule(
        sources=[DeclaredSource(ROWS, stage=Stage.TRAIN)], schema=label_schema(), max_samples=max_samples
    )
    module.setup()
    return module


def kept_paths(module: TableDataModule) -> list[str]:
    dataset = module.dataset(Stage.TRAIN)
    return [str(dataset[index].meta[Sample.CELLS]["image"]) for index in range(len(dataset))]  # the input, by name


def test_an_integer_keeps_that_many_rows() -> None:
    assert len(capped(10).dataset(Stage.TRAIN)) == 10


def test_a_fraction_keeps_that_share_of_rows() -> None:
    assert len(capped(0.25).dataset(Stage.TRAIN)) == 25


def test_a_cap_above_the_table_size_keeps_everything() -> None:
    """A debugging cap must not fail on a table that is already smaller."""
    assert len(capped(500).dataset(Stage.TRAIN)) == 100


def test_rows_are_drawn_at_random_not_taken_from_the_top() -> None:
    """Annotation files often arrive sorted; the head of one is not a sample of it."""
    module = capped(20)

    assert kept_paths(module) != [f"{index}.jpg" for index in range(20)]
    assert {int(module.dataset(Stage.TRAIN)[index].targets["label"]) for index in range(20)} == {0, 1}


def test_the_same_cap_keeps_the_same_rows() -> None:
    """A cap is a debugging aid: the rows it keeps must not move between two runs."""
    assert kept_paths(capped(20)) == kept_paths(capped(20))


def test_a_count_and_a_share_differ_only_by_the_decimal_point() -> None:
    """The sklearn idiom for such arguments: 1 is one row, 1.0 is all of them."""
    assert len(capped(1).dataset(Stage.TRAIN)) == 1
    assert len(capped(1.0).dataset(Stage.TRAIN)) == 100


def test_the_cap_applies_to_each_source_not_to_their_sum() -> None:
    """Combining datasets must not shrink each one to a share of the cap."""
    module = TableDataModule(
        sources=[
            DeclaredSource(labelled(["cat"] * 8), stage=Stage.TRAIN),
            DeclaredSource(labelled(["dog"] * 8), stage=Stage.TRAIN),
        ],
        schema=label_schema(),
        max_samples=4,
    )
    module.setup()

    assert len(module.dataset(Stage.TRAIN)) == 8


def test_a_share_above_one_is_refused_rather_than_read_as_a_count() -> None:
    with pytest.raises(ValueError, match="at most 1.0"):
        TableDataModule(sources=[DeclaredSource(ROWS, stage=Stage.TRAIN)], schema=label_schema(), max_samples=2.0)


def test_a_cap_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        TableDataModule(sources=[DeclaredSource(ROWS, stage=Stage.TRAIN)], schema=label_schema(), max_samples=0)
