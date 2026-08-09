"""The split seed is independent of the global seed — the property a seed sweep needs."""

from __future__ import annotations

import pandas as pd
from lightning import seed_everything

from src.core import Stage
from src.data import random_split

FRACTIONS = {Stage.TRAIN: 0.6, Stage.VAL: 0.2, Stage.TEST: 0.2}


def table() -> pd.DataFrame:
    return pd.DataFrame({"x": range(10)})


def test_the_partition_survives_a_change_of_experiment_seed() -> None:
    """Five runs at five seeds must share one test set, or their metrics do not compare."""
    split = random_split(FRACTIONS, seed=42)

    seed_everything(1, verbose=False)
    first = split(table())[Stage.TEST]["x"].tolist()
    seed_everything(999, verbose=False)
    second = split(table())[Stage.TEST]["x"].tolist()

    assert first == second


def test_changing_the_split_seed_is_how_the_partition_changes() -> None:
    original = random_split(FRACTIONS, seed=42)(table())[Stage.TEST]["x"].tolist()
    other = random_split(FRACTIONS, seed=7)(table())[Stage.TEST]["x"].tolist()

    assert original != other
