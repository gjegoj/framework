"""Stratified splitting: every stage sees the same distribution of a chosen column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core import Stage
from src.data import stratified_split

FRACTIONS = {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}


def shares(table: pd.DataFrame, column: str) -> dict[object, float]:
    counts: dict[object, float] = table[column].value_counts(normalize=True).round(2).to_dict()
    return counts


def test_class_shares_are_preserved_in_every_stage() -> None:
    labels = np.repeat(["cat", "dog", "bird"], [600, 300, 100])
    table = pd.DataFrame({"label": labels})

    parts = stratified_split(FRACTIONS, by="label", seed=42)(table)

    for stage, part in parts.items():
        assert shares(part, "label") == {"cat": 0.6, "dog": 0.3, "bird": 0.1}, stage


def test_integer_labels_are_stratified_as_classes_not_as_a_continuous_range() -> None:
    """Dtype must not decide the strategy: 0/1 labels are classes, not a range to bin.

    Quantile-binning an imbalanced 0/1 column collapses it into a single bin, which
    silently degrades the split to a random one — exactly where stratification matters.
    """
    labels = np.repeat([0, 1], [950, 50])
    table = pd.DataFrame({"label": labels})

    parts = stratified_split(FRACTIONS, by="label", seed=42)(table)

    for stage, part in parts.items():
        assert shares(part, "label") == {0: 0.95, 1: 0.05}, stage


def test_a_continuous_column_is_split_by_quantile_bins() -> None:
    """Distinct values outnumber the bins, so rows are grouped by quantile, not by value."""
    table = pd.DataFrame({"price": np.linspace(0.0, 100.0, 1000)})

    parts = stratified_split(FRACTIONS, by="price", seed=42, bins=5)(table)

    medians = [part["price"].median() for part in parts.values()]
    assert all(abs(median - 50.0) < 5.0 for median in medians), medians


def test_a_value_too_rare_to_spread_goes_to_the_earlier_stage() -> None:
    """Long-tail data must stay usable: one unsplittable row cannot fail the whole run."""
    table = pd.DataFrame({"label": ["cat"] * 500 + ["dog"] * 499 + ["unicorn"]})

    parts = stratified_split(FRACTIONS, by="label", seed=42)(table)

    assert "unicorn" in parts[Stage.TRAIN]["label"].tolist()
    assert sum(len(part) for part in parts.values()) == 1000


def test_every_row_lands_in_exactly_one_stage() -> None:
    table = pd.DataFrame({"label": np.repeat(["a", "b"], 500), "row": range(1000)})

    parts = stratified_split(FRACTIONS, by="label", seed=42)(table)

    assigned = sorted(row for part in parts.values() for row in part["row"])
    assert assigned == list(range(1000))


def test_the_same_seed_gives_the_same_split() -> None:
    table = pd.DataFrame({"label": np.repeat(["a", "b"], 500), "row": range(1000)})
    split = stratified_split(FRACTIONS, by="label", seed=42)

    assert split(table)[Stage.TEST]["row"].tolist() == split(table)[Stage.TEST]["row"].tolist()


def test_an_unknown_column_names_the_available_ones() -> None:
    table = pd.DataFrame({"label": ["a", "b"], "image": ["x.png", "y.png"]})

    with pytest.raises(KeyError, match="image"):
        stratified_split(FRACTIONS, by="labell", seed=42)(table)


def test_too_many_classes_for_the_smallest_stage_explains_the_conflict() -> None:
    """sklearn's raw message is unactionable; ours must name the way out."""
    table = pd.DataFrame({"label": np.repeat([f"class_{index}" for index in range(50)], 2)})

    with pytest.raises(ValueError, match="random_split|at least one row"):
        stratified_split(FRACTIONS, by="label", seed=42)(table)
