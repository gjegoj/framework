"""``build_model``: the seam where model families differ."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.assembly.models import build_model
from src.core import Batch, Stage
from src.models import CompositeModel
from tests.support.configs import TASK, disk_config
from tests.support.entities import profiled
from tests.support.narrowing import tensor


def image_batch(size: int = 2) -> Batch:
    return Batch(
        inputs={"image": torch.randn(size, 3, 32, 32)},
        targets={"label": torch.zeros(size, dtype=torch.long)},
    )


def test_the_model_and_its_tasks_come_back_together(dataset_root: Path) -> None:
    model, tasks = build_model(disk_config(dataset_root), profiled())

    assert isinstance(model, CompositeModel)
    assert [task.name for task in tasks] == ["label"]


def test_heads_are_sized_from_profiled_facts(dataset_root: Path) -> None:
    """Output sizes come from the data, never from config."""
    model, _ = build_model(disk_config(dataset_root), profiled(classes=5))

    assert tensor(model.predict(image_batch()).outputs["label"]).shape == (2, 5)


def test_every_stage_gets_its_own_metric_set(dataset_root: Path) -> None:
    _, tasks = build_model(disk_config(dataset_root), profiled())

    assert set(tasks[0].metrics) == set(Stage)


def test_a_configured_loss_replaces_the_objectives_default(dataset_root: Path) -> None:
    """Same criterion, different knob: smoothing must actually reach the loss."""
    tasks = {"label": TASK | {"loss": {"name": "cross_entropy", "label_smoothing": 0.4}}}
    torch.manual_seed(0)
    smoothed, _ = build_model(disk_config(dataset_root, tasks=tasks), profiled())
    torch.manual_seed(0)
    plain, _ = build_model(disk_config(dataset_root), profiled())

    batch = image_batch()
    assert smoothed.step(batch).loss.total.item() != plain.step(batch).loss.total.item()


def test_a_task_may_ask_for_the_backbones_native_head(dataset_root: Path) -> None:
    """The native head is timm's own classifier, wrapped — not our LinearHead."""
    tasks = {"label": TASK | {"native_head": True}}

    model, _ = build_model(disk_config(dataset_root, tasks=tasks), profiled())

    assert any("heads.label._module" in key for key in model.state_dict())
    assert tensor(model.predict(image_batch()).outputs["label"]).shape == (2, 2)


def test_an_unregistered_model_name_is_rejected(dataset_root: Path) -> None:
    with pytest.raises(LookupError, match="timm"):
        build_model(disk_config(dataset_root, model={"name": "nope"}), profiled())
