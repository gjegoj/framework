"""``max_samples`` shrinks a run wherever the source sits, without a rule to remember."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.assembly.data import build_data_module, build_splitter
from src.config.data import SplitConfig
from src.core import DataProfile, Stage
from tests.support.configs import INPUTS, SPLIT, paper_config

FRACTIONS = SPLIT


def write_rows(path: Path, count: int) -> None:
    pd.DataFrame(
        {
            "image": [f"{index}.png" for index in range(count)],
            "label": ["cat", "dog"] * (count // 2),
        }
    ).to_csv(path, index=False)


def experiment(**data: Any) -> Any:
    """The base experiment over the sources a test just wrote, and nothing it did not name."""
    return paper_config(data={"inputs": INPUTS} | data)


def stage_sizes(module: Any) -> dict[Stage, int]:
    module.setup(DataProfile())
    return {stage: len(module.dataset(stage)) for stage in Stage}


def test_one_source_is_capped_before_the_split(tmp_path: Path) -> None:
    write_rows(tmp_path / "all.csv", 100)
    config = experiment(source=str(tmp_path / "all.csv"), split=FRACTIONS, max_samples=20)

    assert stage_sizes(build_data_module(config)) == {Stage.TRAIN: 10, Stage.VAL: 5, Stage.TEST: 5}


def test_per_stage_sources_are_capped_one_stage_at_a_time(tmp_path: Path) -> None:
    for stage in ("train", "val", "test"):
        write_rows(tmp_path / f"{stage}.csv", 100)
    config = experiment(
        source={stage: str(tmp_path / f"{stage}.csv") for stage in ("train", "val", "test")},
        max_samples=20,
    )

    assert stage_sizes(build_data_module(config)) == {Stage.TRAIN: 20, Stage.VAL: 20, Stage.TEST: 20}


def test_a_share_caps_the_same_way(tmp_path: Path) -> None:
    write_rows(tmp_path / "all.csv", 100)
    config = experiment(source=str(tmp_path / "all.csv"), split=FRACTIONS, max_samples=0.4)

    assert sum(stage_sizes(build_data_module(config)).values()) == 40


@pytest.mark.parametrize(
    ("declared", "expected"),
    [({}, "random"), ({"stratify_by": "label"}, "stratified"), ({"group_by": "patient"}, "group")],
)
def test_the_declared_column_picks_the_splitter(declared: dict[str, str], expected: str) -> None:
    splitter = build_splitter(SplitConfig(train=0.5, val=0.25, test=0.25, **declared))

    assert expected in getattr(splitter, "__qualname__", "")
