"""The form of ``data.source`` decides how stages are obtained, and the config says so."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config.data import DataConfig
from src.core import Stage

INPUTS = {"image": {"column": "path", "loader": "image"}}
FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}


def config(**overrides: Any) -> DataConfig:
    return DataConfig.model_validate({"inputs": INPUTS, **overrides})


def paths_per_stage(data: DataConfig) -> dict[Stage, list[str | list[str]]]:
    """The files each stage declares, however many sources it draws on."""
    declared = data.source
    assert isinstance(declared, dict)
    return {
        stage: [source.path for source in (sources if isinstance(sources, list) else [sources])]
        for stage, sources in declared.items()
    }


def test_per_stage_sources_are_accepted_without_a_split() -> None:
    data = config(source={"train": "train.csv", "val": "val.csv", "test": "test.csv"})

    assert data.split is None
    assert paths_per_stage(data) == {
        Stage.TRAIN: ["train.csv"],
        Stage.VAL: ["val.csv"],
        Stage.TEST: ["test.csv"],
    }


def test_a_stage_may_draw_on_several_sources() -> None:
    data = config(source={"train": ["a.csv", "b.csv"], "val": "val.csv"})

    assert paths_per_stage(data) == {Stage.TRAIN: ["a.csv", "b.csv"], Stage.VAL: ["val.csv"]}


def test_per_stage_sources_without_train_are_refused() -> None:
    with pytest.raises(ValidationError, match="need a 'train' entry"):
        config(source={"val": "val.csv", "test": "test.csv"})


def test_an_unknown_stage_name_is_refused() -> None:
    with pytest.raises(ValidationError):
        config(source={"train": "train.csv", "holdout": "holdout.csv"})


def test_a_single_source_with_a_split_still_works() -> None:
    data = config(source="annotations.csv", split=FRACTIONS)

    assert data.split is not None
    assert data.split.fractions()[Stage.TRAIN] == 0.6


def test_stratifying_and_grouping_at_once_is_refused() -> None:
    """One moves single rows to balance classes, the other forbids moving them apart."""
    with pytest.raises(ValidationError, match="cannot be combined"):
        config(source="a.csv", split={**FRACTIONS, "stratify_by": "label", "group_by": "patient"})


def test_max_samples_accepts_a_count_and_a_share() -> None:
    assert config(source="a.csv", split=FRACTIONS, max_samples=100).max_samples == 100
    assert config(source="a.csv", split=FRACTIONS, max_samples=0.1).max_samples == 0.1


def test_a_max_samples_of_zero_is_refused() -> None:
    with pytest.raises(ValidationError):
        config(source="a.csv", split=FRACTIONS, max_samples=0)
