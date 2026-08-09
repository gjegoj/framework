"""Capping rows: the same declaration shrinks a run wherever the source sits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import InMemorySource, LimitedSource

TABLE = pd.DataFrame({"row": range(100), "label": ["cat", "dog"] * 50})


def test_an_integer_keeps_that_many_rows() -> None:
    assert len(LimitedSource(InMemorySource(TABLE), 10).read()) == 10


def test_a_fraction_keeps_that_share_of_rows() -> None:
    assert len(LimitedSource(InMemorySource(TABLE), 0.25).read()) == 25


def test_a_cap_above_the_table_size_keeps_everything() -> None:
    """A debugging cap must not fail on a table that is already smaller."""
    assert len(LimitedSource(InMemorySource(TABLE), 500).read()) == 100


def test_rows_are_drawn_at_random_not_taken_from_the_top() -> None:
    """Annotation files often arrive sorted; the head of one is not a sample of it."""
    kept = LimitedSource(InMemorySource(TABLE), 20).read()

    assert kept["row"].tolist() != list(range(20))
    assert kept["label"].nunique() == 2


def test_the_same_seed_keeps_the_same_rows() -> None:
    first = LimitedSource(InMemorySource(TABLE), 20, seed=7).read()
    second = LimitedSource(InMemorySource(TABLE), 20, seed=7).read()

    assert first["row"].tolist() == second["row"].tolist()


def test_a_count_and_a_share_differ_only_by_the_decimal_point() -> None:
    """The sklearn idiom for such arguments: 1 is one row, 1.0 is all of them."""
    assert len(LimitedSource(InMemorySource(TABLE), 1).read()) == 1
    assert len(LimitedSource(InMemorySource(TABLE), 1.0).read()) == 100


def test_a_share_above_one_is_refused_rather_than_read_as_a_count() -> None:
    with pytest.raises(ValueError, match="at most 1.0"):
        LimitedSource(InMemorySource(TABLE), 2.0)


def test_a_cap_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        LimitedSource(InMemorySource(TABLE), 0)
