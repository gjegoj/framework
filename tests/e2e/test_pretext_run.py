"""A run whose supervision nothing annotated: the augmentation draws the answer as it draws the picture.

Every seam between the pixel pipeline and the loop is on this path — the geometries the encoders
publish, the binding that lets one target through a pipeline that otherwise carries only pixels, the
vocabulary the table could never have taught, and the collation that has to find the drawn value where
an annotated one would have been.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.experiment import run
from src.transforms.augmentations import QUARTER_TURNS
from tests.support.declarations import SIZE, pixel_pipeline, smallest_run
from tests.support.table import write_table

TURNS = {index: str(index) for index in range(QUARTER_TURNS)}


def turning() -> dict[str, Any]:
    """The shipped pixel chain with the rotation in front of it, where a size-changing step belongs."""
    chain = pixel_pipeline(SIZE)
    # ``turn`` is the task, ``angle`` is the column it reads: a sample's targets are keyed by the
    # first, and by the time a transform sees one the second is the data layer's own business.
    return {**chain, "transforms": [{"_target_": "src.transforms.Rotate90", "task": "turn"}, *chain["transforms"]]}


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Wider than the other runs need: four outcomes are drawn per sample, and a split of two rows
    would land on one of them often enough to fail a run that is working."""
    return write_table(tmp_path_factory.mktemp("upright"), samples=32)


@pytest.fixture
def declared(table: Path, tmp_path: Path) -> Mapping[str, Any]:
    """One task, and the only thing that knows its answers is the augmentation that writes them.

    Every stage declares the rotation, evaluation included: the answer lives in the pipeline, so a
    stage that leaves the augmentation out has nothing left to measure but a column of zeroes.
    """
    return {
        **smallest_run(table, tmp_path / "run"),
        "transforms": dict.fromkeys(("train", "val", "test"), turning()),
        "tasks": {"turn": {"kind": "classification", "target_column": "angle", "classes": TURNS}},
    }


def drawn(loader: Any) -> torch.Tensor:
    return torch.cat([batch.targets["turn"] for batch in loader])


def test_the_turn_the_augmentation_drew_is_what_the_batch_carries(declared: Mapping[str, Any]) -> None:
    """The column is upright throughout, so every other value in the batch was drawn rather than read."""
    built = build(load_config(declared))

    turns = drawn(built.data.train_dataloader())

    assert turns.unique().numel() > 1
    assert set(turns.tolist()) <= set(TURNS)


def test_evaluation_is_answered_by_its_own_draw_too(declared: Mapping[str, Any]) -> None:
    built = build(load_config(declared))

    assert drawn(built.data.val_dataloader()).unique().numel() > 1


def test_the_run_trains_and_reports_on_a_task_nothing_annotated(declared: Mapping[str, Any]) -> None:
    built = build(load_config(declared))

    run(built)

    assert {"test/loss", "test/turn/cross_entropy"} <= set(built.trainer.callback_metrics)
