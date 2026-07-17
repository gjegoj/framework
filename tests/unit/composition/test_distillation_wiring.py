"""Distillation wiring: teacher loading, brick resolution, and the build_lit_module branch."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.composition.wiring import build_lit_module, build_tasks, build_teachers
from src.composition.wiring.training import resolve_distillation_bricks
from src.config import load_config
from src.config.schema import DistillationConfig, ExperimentConfig
from src.core.entities import Task
from src.core.runtime import RuntimeContext
from src.losses.distillation import KLDivergenceCriterion
from src.models import build_composite_model
from src.models.assembly import CompositeModel
from src.models.registry import backbones
from src.training import DistillationLitModule, LitModule, OptimizerBuilder
from tests.support.builders import minimal_config as _minimal_config


def _teacher_ckpt(path: Path, *, lightning_prefix: bool) -> str:
    """Save a resnet18+classification-head state dict; optionally under the LitModule 'model.' prefix."""
    runtime = RuntimeContext(num_classes={"label": 3})
    tasks = build_tasks(load_config(_minimal_config()), runtime)
    backbone = backbones.create("timm", name="resnet18", pretrained=False)
    model = build_composite_model(backbone, {task.name: task.head_spec for task in tasks})
    state = model.state_dict()
    if lightning_prefix:
        state = {f"model.{key}": value for key, value in state.items()}
    torch.save({"state_dict": state}, path)
    return str(path)


def _distillation_config(tmp_path: Path, **distillation_overrides: object) -> ExperimentConfig:
    ckpt = _teacher_ckpt(tmp_path / "teacher.ckpt", lightning_prefix=True)
    raw = _minimal_config()
    raw["distillation"] = {
        "teachers": [{"backbone": {"name": "resnet18", "pretrained": False}, "ckpt_path": ckpt}],
        **distillation_overrides,
    }
    return load_config(raw)


def _section(config: ExperimentConfig) -> DistillationConfig:
    """Narrow ``config.distillation`` to non-None for the type checker."""
    assert config.distillation is not None
    return config.distillation


class TestBuildTeachers:
    @pytest.mark.parametrize("lightning_prefix", [True, False], ids=["lightning_ckpt", "raw_state_dict"])
    def test_loads_teacher_from_either_format(self, tmp_path: Path, lightning_prefix: bool) -> None:
        ckpt = _teacher_ckpt(tmp_path / "teacher.ckpt", lightning_prefix=lightning_prefix)
        raw = _minimal_config()
        raw["distillation"] = {"teachers": [{"backbone": {"name": "resnet18", "pretrained": False}, "ckpt_path": ckpt}]}
        config = load_config(raw)
        tasks = build_tasks(config, RuntimeContext(num_classes={"label": 3}))

        ensemble = build_teachers(_section(config), tasks)

        assert all(not parameter.requires_grad for parameter in ensemble.parameters())
        soft = ensemble({"image": torch.randn(2, 3, 64, 64)})
        assert soft["label"].shape == (2, 3)


class TestResolveDistillationBricks:
    def _tasks(self) -> list[Task]:
        return build_tasks(load_config(_minimal_config()), RuntimeContext(num_classes={"label": 3}))

    def test_default_loss_is_kl_with_configured_temperature(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, temperature=3.0)
        criteria, weights = resolve_distillation_bricks(_section(config), self._tasks())
        assert isinstance(criteria["label"], KLDivergenceCriterion)
        assert criteria["label"].temperature == 3.0
        assert weights == {"label": 1.0}

    def test_scalar_weight_applies_to_all_tasks(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, weight=0.7)
        _, weights = resolve_distillation_bricks(_section(config), self._tasks())
        assert weights == {"label": 0.7}

    def test_per_task_weight_map(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, weight={"label": 0.4})
        _, weights = resolve_distillation_bricks(_section(config), self._tasks())
        assert weights == {"label": 0.4}

    def test_explicit_loss_spec_wins(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, temperature=3.0, loss={"name": "kl_divergence", "temperature": 5.0})
        criteria, _ = resolve_distillation_bricks(_section(config), self._tasks())
        assert criteria["label"].temperature == 5.0

    def test_unknown_task_rejected(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, tasks=["nope"])
        with pytest.raises(ValueError, match="unknown task"):
            resolve_distillation_bricks(_section(config), self._tasks())


class TestBuildLitModuleBranch:
    def _model_and_tasks(self, config: ExperimentConfig) -> tuple[CompositeModel, list[Task]]:
        tasks = build_tasks(config, RuntimeContext(num_classes={"label": 3}))
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {task.name: task.head_spec for task in tasks})
        return model, tasks

    def test_without_distillation_builds_plain_lit_module(self) -> None:
        config = load_config(_minimal_config())
        model, tasks = self._model_and_tasks(config)
        lit = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert isinstance(lit, LitModule)
        assert not isinstance(lit, DistillationLitModule)

    def test_with_distillation_builds_distillation_module(self, tmp_path: Path) -> None:
        config = _distillation_config(tmp_path, weight=0.6)
        model, tasks = self._model_and_tasks(config)
        lit = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert isinstance(lit, DistillationLitModule)
        assert lit._distillation_weights == {"label": 0.6}
