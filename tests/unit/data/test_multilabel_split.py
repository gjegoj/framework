"""Stratifying a column whose cells hold several labels at once."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core import Stage
from src.data import stratified_split

FRACTIONS = {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}


def tagged(rows: int = 600) -> pd.DataFrame:
    """Rows tagged from a vocabulary of five, with deliberately uneven label rates."""
    rng = np.random.default_rng(0)
    rates = {"common": 0.7, "frequent": 0.4, "middling": 0.2, "rare": 0.08, "scarce": 0.03}
    tags = []
    for _ in range(rows):
        present = [label for label, rate in rates.items() if rng.random() < rate]
        tags.append(",".join(present))
    return pd.DataFrame({"tags": tags, "row": range(rows)})


def rate_of(label: str, table: pd.DataFrame) -> float:
    return float(table["tags"].str.split(",").apply(lambda tags: label in tags).mean())


def test_every_label_keeps_its_rate_in_every_stage() -> None:
    """The property that matters: each label, not each combination, stays proportional."""
    table = tagged()

    parts = stratified_split(FRACTIONS, by="tags", seed=42)(table)

    for label in ("common", "frequent", "middling", "rare", "scarce"):
        overall = rate_of(label, table)
        for stage, part in parts.items():
            assert abs(rate_of(label, part) - overall) < 0.06, f"{label} drifted in {stage}"


def test_treating_combinations_as_classes_would_not_survive_this_table() -> None:
    """Why a separate algorithm: with five labels most combinations occur a handful of times."""
    table = tagged()

    assert table["tags"].nunique() > 20


def test_every_row_lands_in_exactly_one_stage() -> None:
    parts = stratified_split(FRACTIONS, by="tags", seed=42)(tagged())

    assigned = sorted(row for part in parts.values() for row in part["row"])
    assert assigned == list(range(600))


def test_the_same_seed_gives_the_same_split() -> None:
    split = stratified_split(FRACTIONS, by="tags", seed=42)

    assert split(tagged())[Stage.TEST]["row"].tolist() == split(tagged())[Stage.TEST]["row"].tolist()


def test_the_split_does_not_disturb_the_runs_own_randomness() -> None:
    """The library behind this reads numpy's global RNG; borrowing it must leave no trace."""
    np.random.seed(1234)
    expected = np.random.rand(3).tolist()

    np.random.seed(1234)
    stratified_split(FRACTIONS, by="tags", seed=42)(tagged())

    assert np.random.rand(3).tolist() == expected


def test_a_single_label_column_still_uses_the_exact_strategy() -> None:
    """One label per row needs no approximation, so the ordinary path must stay in charge."""
    table = pd.DataFrame({"label": np.repeat(["cat", "dog"], [500, 100])})

    parts = stratified_split(FRACTIONS, by="label", seed=42)(table)

    for part in parts.values():
        assert round(part["label"].value_counts(normalize=True)["cat"], 2) == 0.83


def test_lists_in_the_column_are_understood_too() -> None:
    rng = np.random.default_rng(1)
    vocabulary = ["a", "b", "c"]
    table = pd.DataFrame(
        {"tags": [[label for label in vocabulary if rng.random() < 0.5] for _ in range(300)], "row": range(300)}
    )

    parts = stratified_split(FRACTIONS, by="tags", seed=42)(table)

    assert sum(len(part) for part in parts.values()) == 300
