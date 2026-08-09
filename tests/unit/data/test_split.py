"""``Splitter`` contract: deterministic, exhaustive, disjoint stage splits."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import SplitConfig
from src.core import Stage
from src.data import random_split


def make_table(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"path": [f"img_{index}.jpg" for index in range(rows)]})


def test_random_split_respects_fractions() -> None:
    split = random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}, seed=42)

    parts = split(make_table(8))

    assert {stage: len(part) for stage, part in parts.items()} == {
        Stage.TRAIN: 4,
        Stage.VAL: 2,
        Stage.TEST: 2,
    }


def test_a_run_declaring_no_test_share_cuts_no_test_rows() -> None:
    """The whole reason a zero share is dropped rather than passed on as zero.

    The last stage a splitter cuts takes whatever flooring left over, and test is
    last. Measured on these very numbers: `{train: 0.7, val: 0.3, test: 0.0}` handed
    over intact splits nine rows 6/2/**1** — so a run that declared no test set
    would report a test metric computed on that one leftover row, and val would be
    short of it. Dropped, the same nine rows split 6/3.
    """
    declared = SplitConfig.model_validate({"train": 0.7, "val": 0.3, "test": 0})

    parts = random_split(declared.fractions(), seed=declared.seed)(make_table(9))

    assert {stage: len(part) for stage, part in parts.items()} == {Stage.TRAIN: 6, Stage.VAL: 3}


def test_random_split_covers_every_row_exactly_once() -> None:
    split = random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.5}, seed=0)
    table = make_table(10)

    parts = split(table)

    covered = sorted(path for part in parts.values() for path in part["path"])
    assert covered == sorted(table["path"])


def test_random_split_is_deterministic_for_a_seed() -> None:
    fractions = {Stage.TRAIN: 0.75, Stage.VAL: 0.25}

    first = random_split(fractions, seed=7)(make_table(12))
    second = random_split(fractions, seed=7)(make_table(12))

    assert first[Stage.TRAIN]["path"].tolist() == second[Stage.TRAIN]["path"].tolist()


def test_random_split_gives_the_remainder_to_the_last_stage() -> None:
    split = random_split({Stage.TRAIN: 1 / 3, Stage.VAL: 2 / 3}, seed=0)

    parts = split(make_table(4))

    assert len(parts[Stage.TRAIN]) == 1
    assert len(parts[Stage.VAL]) == 3


def test_random_split_rejects_fractions_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum"):
        random_split({Stage.TRAIN: 0.5, Stage.VAL: 0.2}, seed=0)


def test_random_split_rejects_an_empty_plan() -> None:
    with pytest.raises(ValueError, match="empty"):
        random_split({}, seed=0)
