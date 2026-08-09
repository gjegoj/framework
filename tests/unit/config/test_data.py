"""``DataConfig`` and ``SplitConfig``: sources, inputs, and split fractions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import DataConfig, SplitConfig
from src.config.data import SourceConfig
from src.core import Stage


def make_raw_data() -> dict[str, object]:
    return {
        "source": "annotations.csv",
        "inputs": {"image": {"column": "path", "loader": "image"}},
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
    }


def test_a_valid_data_section_parses() -> None:
    data = DataConfig.model_validate(make_raw_data())

    assert isinstance(data.source, SourceConfig)
    assert data.source.path == "annotations.csv"
    assert data.inputs["image"].column == "path"
    assert data.inputs["image"].loader.name == "image"


def test_a_list_declares_several_sources_to_combine() -> None:
    """Each is divided by the same fractions, so every one reaches every stage."""
    raw = make_raw_data() | {"source": ["a.csv", "b.csv"]}

    sources = DataConfig.model_validate(raw).source

    assert isinstance(sources, list)
    assert [source.path for source in sources] == ["a.csv", "b.csv"]


def test_one_source_may_still_span_several_files() -> None:
    """Files of one dataset are concatenated; that is a different thing from two datasets."""
    raw = make_raw_data() | {"source": {"path": ["part1.csv", "part2.csv"]}}

    source = DataConfig.model_validate(raw).source

    assert isinstance(source, SourceConfig)
    assert source.path == ["part1.csv", "part2.csv"]


def test_a_section_with_no_inputs_is_a_valid_declaration() -> None:
    """A vendor pipeline reads its images from its own descriptor and declares no columns.

    That a *table* needs at least one is true of the table, and is asserted where the
    table's schema is built (`tests/unit/data/test_schema.py`). Stated here as well, it
    would forbid a whole model family from ever being declared.
    """
    raw = make_raw_data() | {"inputs": {}}

    assert DataConfig.model_validate(raw).inputs == {}


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum"):
        SplitConfig.model_validate({"train": 0.5, "val": 0.2, "test": 0.2})


def test_split_exposes_stage_keyed_fractions() -> None:
    split = SplitConfig.model_validate({"train": 0.5, "val": 0.25, "test": 0.25})

    assert split.fractions() == {Stage.TRAIN: 0.5, Stage.VAL: 0.25, Stage.TEST: 0.25}
    assert split.seed == 42


def test_a_zero_test_share_leaves_the_stage_out_rather_than_asking_for_none_of_it() -> None:
    """A splitter gives its *last* stage the rounding remainder, and test is last.

    Handed on as `0.0`, the stage would still be cut — and would take the one or two
    rows that flooring left over. A run declaring no test set would then report a
    test metric computed on two samples, which is worse than either having one or
    not. Left out, the stage does not exist, and that is an answer `TrainingData`
    already acts on: it tests on val and says so.
    """
    split = SplitConfig.model_validate({"train": 0.8, "val": 0.2, "test": 0})

    assert split.fractions() == {Stage.TRAIN: 0.8, Stage.VAL: 0.2}


@pytest.mark.parametrize(
    "declared",
    [
        {"train": 0.0, "val": 1.0, "test": 0.0},  # sums to 1, so only the bound can refuse it
        {"train": 1.0, "val": 0.0, "test": 0.0},
    ],
)
def test_train_and_val_may_not_be_zeroed_the_way_test_may(declared: dict[str, float]) -> None:
    """Encoders are fitted on train, and val is what a zero test share falls back to.

    Both cases sum to 1, so the sum rule cannot be what refuses them — the bound is.
    """
    with pytest.raises(ValidationError, match="greater than 0"):
        SplitConfig.model_validate(declared)
