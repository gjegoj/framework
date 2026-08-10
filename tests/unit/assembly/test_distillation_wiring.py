"""The distillation section, turned into a student wearing its teachers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from src.assembly.models import backbone_path, build_model
from src.core.entities import Batch
from src.losses import KLDivergenceCriterion
from src.models import CompositeModel, DistilledModel, without_teachers
from tests.support.configs import disk_config
from tests.support.entities import profiled

if TYPE_CHECKING:
    from pathlib import Path

TEACHER = {"name": "timm", "model_name": "resnet18", "pretrained": False}


def one_batch() -> Batch:
    return Batch(inputs={"image": torch.randn(2, 3, 16, 16)}, targets={"label": torch.zeros(2, dtype=torch.long)})


def _distilled(root: Path, loss: dict[str, Any]) -> DistilledModel:
    declared = {"teachers": [{"backbone": TEACHER}], "loss": loss}
    model, _ = build_model(disk_config(root, distillation=declared), profiled())
    assert isinstance(model, DistilledModel)
    return model


def test_no_distillation_section_builds_the_student_alone(dataset_root: Path) -> None:
    """The default has to stay ordinary training; a teacher is opted into."""
    model, _ = build_model(disk_config(dataset_root), profiled())

    assert isinstance(model, CompositeModel)


def test_a_declared_teacher_wraps_the_student(dataset_root: Path) -> None:
    """What arrives is the student it was, wearing the signal it will be judged against."""
    declared = {"teachers": [{"backbone": TEACHER}]}

    model, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())

    assert isinstance(model, DistilledModel)
    assert isinstance(model.student, CompositeModel)
    assert len(model.teachers) == 1


def test_the_soft_terms_declared_weight_reaches_the_criterion_built_from_it(dataset_root: Path) -> None:
    """The weight rides on the loss declaration, the way every other term in this framework carries its own."""
    logits, target = torch.randn(2, 2), torch.randn(2, 2)

    plain = _distilled(dataset_root, {"name": "kl_divergence"})
    weighted = _distilled(dataset_root, {"name": "kl_divergence", "weight": 0.5})

    unscaled = plain.criterion(logits, target).total.item()
    assert weighted.criterion(logits, target).total.item() == pytest.approx(0.5 * unscaled)


def test_the_comparison_defaults_to_temperature_scaled_kl(dataset_root: Path) -> None:
    """Declaring teachers and nothing else is the ordinary way to reach for distillation."""
    model, _ = build_model(disk_config(dataset_root, distillation={"teachers": [{"backbone": TEACHER}]}), profiled())

    assert isinstance(model, DistilledModel)
    assert isinstance(model.criterion, KLDivergenceCriterion)


def test_the_teacher_carries_the_students_task_shapes(dataset_root: Path) -> None:
    """Their logits are compared, so they have to be the same shape by construction, not by luck."""
    declared = {"teachers": [{"backbone": TEACHER}]}
    model, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())
    assert isinstance(model, DistilledModel)

    batch = one_batch()
    student = model.student.predict(batch).logits
    teacher = model.teachers[0].predict(batch).logits

    assert student is not None
    assert teacher is not None
    assert student["label"].shape == teacher["label"].shape


def test_a_teacher_is_loaded_from_a_runs_checkpoint(dataset_root: Path, tmp_path: Path) -> None:
    """Measured: a run's keys are the model's own under a `model.` prefix, so loading is a strip."""
    source, _ = build_model(disk_config(dataset_root), profiled())
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({"state_dict": {f"model.{name}": value for name, value in source.state_dict().items()}}, checkpoint)
    declared = {"teachers": [{"backbone": TEACHER, "checkpoint_path": str(checkpoint)}]}

    model, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())

    assert isinstance(model, DistilledModel)
    loaded = model.teachers[0].state_dict()
    assert all(torch.allclose(loaded[name], value) for name, value in source.state_dict().items())


def test_a_teacher_without_a_checkpoint_keeps_the_weights_it_was_built_with(dataset_root: Path) -> None:
    """A backbone whose pretrained weights are already the teacher must not need a file to say so."""
    declared = {"teachers": [{"backbone": TEACHER}]}
    built, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())
    assert isinstance(built, DistilledModel)

    again, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())

    assert isinstance(again, DistilledModel)
    assert len(again.teachers) == 1


def test_the_artifact_is_written_from_the_student(dataset_root: Path) -> None:
    """Measured: a traced graph carries an unused teacher, so export must be handed the student."""
    declared = {"teachers": [{"backbone": TEACHER}]}
    model, _ = build_model(disk_config(dataset_root, distillation=declared), profiled())

    assert isinstance(model, DistilledModel)
    assert without_teachers(model) is model.student


def test_a_distilled_runs_backbone_path_names_the_nested_student(dataset_root: Path) -> None:
    """A freeze callback declares this path in config, and distillation moves what it points at."""
    declared = {"teachers": [{"backbone": TEACHER}]}

    assert backbone_path(disk_config(dataset_root)) == "model.backbone"
    assert backbone_path(disk_config(dataset_root, distillation=declared)) == "model.student.backbone"


def test_the_adapters_versus_freeze_guard_still_fires_in_a_distilled_run(dataset_root: Path) -> None:
    """A guard that silently stops matching is worse than none: its failure is a loss that never moves."""
    config = disk_config(
        dataset_root,
        model={"name": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16},
        adapters={"name": "lora", "target_modules": ["fc1"], "rank": 4},
        distillation={"teachers": [{"backbone": TEACHER}]},
        callbacks=[{"name": "freeze", "modules": ["model.student.backbone"]}],
    )

    with pytest.raises(ValueError, match="model.student.backbone"):
        build_model(config, profiled())
