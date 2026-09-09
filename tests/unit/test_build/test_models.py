"""``build_model``: a backbone is composed with one head per task; a model is built as it is."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.build import build, build_data_module, build_kinds
from src.core import Batch, Stage, TaskFacts
from src.models import CompositeModel
from tests.support.configs import TASK, disk_config, metrics_of, model_of
from tests.support.detection import annotation_tree, detection_config
from tests.support.entities import dataset_facts
from tests.support.fakes import OwnLossModel
from tests.support.narrowing import tensor


def image_batch(size: int = 2) -> Batch:
    return Batch(
        inputs={"image": torch.randn(size, 3, 32, 32)},
        targets={"label": torch.zeros(size, dtype=torch.long)},
    )


def test_the_model_and_its_tasks_come_back_together(dataset_root: Path) -> None:
    model, tasks = model_of(disk_config(dataset_root), dataset_facts())

    assert isinstance(model, CompositeModel)
    assert [task.name for task in tasks] == ["label"]


def test_heads_are_sized_from_profiled_facts(dataset_root: Path) -> None:
    """Output sizes come from the data, never from config."""
    model, _ = model_of(disk_config(dataset_root), dataset_facts(label=5))

    assert tensor(model.predict(image_batch()).outputs["label"]).shape == (2, 5)


def test_every_stage_gets_its_own_metric_set(dataset_root: Path) -> None:
    metrics = metrics_of(disk_config(dataset_root), dataset_facts())

    assert set(metrics["label"]) == set(Stage)


def test_a_configured_loss_replaces_the_objectives_default(dataset_root: Path) -> None:
    """Same criterion, different knob: smoothing must actually reach the loss."""
    tasks = {"label": TASK | {"loss": {"name": "cross_entropy", "label_smoothing": 0.4}}}
    torch.manual_seed(0)
    smoothed, _ = model_of(disk_config(dataset_root, tasks=tasks), dataset_facts())
    torch.manual_seed(0)
    plain, _ = model_of(disk_config(dataset_root), dataset_facts())

    batch = image_batch()
    assert smoothed.step(batch).loss.total.item() != plain.step(batch).loss.total.item()


def test_a_task_may_ask_for_the_backbones_native_head(dataset_root: Path) -> None:
    """``native`` is a reserved name, never a registered head: the backbone's own classifier serves, as it is."""
    tasks = {"label": TASK | {"head": {"name": "native"}}}

    model, _ = model_of(disk_config(dataset_root, tasks=tasks), dataset_facts())

    assert {"heads.label.weight", "heads.label.bias"} <= set(model.state_dict())
    assert tensor(model.predict(image_batch()).outputs["label"]).shape == (2, 2)


def test_an_unregistered_model_name_is_rejected(dataset_root: Path) -> None:
    with pytest.raises(LookupError, match="timm"):
        model_of(disk_config(dataset_root, model={"name": "nope"}), dataset_facts())


def test_a_task_without_a_target_column_gets_facts_without_any(dataset_root: Path) -> None:
    """Metric learning reads no column, so setup learned nothing about it — and that is a value, not an error."""
    tasks = {"label": TASK, "embedding": {"kind": "ranking", "streams": "features"}}

    _, built = model_of(disk_config(dataset_root, tasks=tasks), dataset_facts())

    assert {task.name: task.facts for task in built}["embedding"] == TaskFacts()


WHOLE = {"_target_": "tests.support.fakes.OwnLossModel", "num_classes": 2}
"""A model that arrives whole: nothing composes it, so what it needs it declares."""


def test_a_model_reached_by_target_is_built_as_it_is(dataset_root: Path) -> None:
    """A whole model owns its head and its loss; the tasks still come back, for the metrics and the report."""
    model, tasks = model_of(disk_config(dataset_root, model=WHOLE), dataset_facts())

    assert isinstance(model, OwnLossModel)
    assert [task.name for task in tasks] == ["label"]


def test_adapters_on_a_model_that_is_not_composed_are_refused_by_name(dataset_root: Path) -> None:
    """Adapters reparameterize a backbone this framework composed; a whole model has none to offer."""
    lora = {"name": "lora", "target_modules": ["classifier"], "rank": 2}

    with pytest.raises(ValueError, match=r"'adapters'.*OwnLossModel"):
        model_of(disk_config(dataset_root, model=WHOLE, adapters=lora), dataset_facts())


def test_distillation_over_a_model_that_is_not_composed_is_refused_by_name(dataset_root: Path) -> None:
    """Distillation compares per-task logits the composite family exposes; a whole model is not it."""
    teacher = {"name": "timm", "model_name": "resnet18", "pretrained": False}
    declared = {"teachers": [{"backbone": teacher}], "loss": {"name": "kl_divergence"}}

    with pytest.raises(ValueError, match=r"'distillation'.*OwnLossModel"):
        model_of(disk_config(dataset_root, model=WHOLE, distillation=declared), dataset_facts())


def test_a_target_that_is_neither_a_backbone_nor_a_model_is_refused_by_name(dataset_root: Path) -> None:
    """The model section builds one of two things, and anything else dies naming both."""
    with pytest.raises(TypeError, match=r"Backbone.*Model"):
        model_of(disk_config(dataset_root, model={"_target_": "torch.nn.Identity"}), dataset_facts())


def test_a_composed_detection_model_builds_and_is_refused_training_by_name(tmp_path: Path) -> None:
    """Stage 2: the model builds (``build_model`` keeps that promise), and the run refuses to train it."""
    annotation_tree(tmp_path)
    config = detection_config(tmp_path, model={"name": "ultralytics", "model_name": "yolov8n.yaml"})
    facts = build_data_module(config, build_kinds(config)).setup()
    model, _ = model_of(config, facts)
    assert isinstance(model, CompositeModel)

    with pytest.raises(ValueError, match="no criterion yet"):
        build(config)
