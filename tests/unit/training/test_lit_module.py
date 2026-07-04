"""LitModule scaffolding: loss aggregation, criterion-parameter training, step contract, verbosity."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult, Task
from src.core.enums import Stage
from src.core.ports import Criterion
from src.models import build_composite_model
from src.tasks import classification
from src.training import (
    LitModule,
    OptimizerBuilder,
    WeightedSumAggregator,
)


def _loss(value: float, name: str = "ce") -> LossResult:
    t = torch.tensor(value)
    return LossResult(total=t, components={name: t})


class _ScaledCrossEntropy(Criterion):
    """A criterion carrying one learnable scalar — stands in for InfoNCE/SigLIP's logit_scale."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        value = F.cross_entropy(logits * self.logit_scale.exp(), target.long())
        return LossResult(total=value, components={"scaled_ce": value})


class TestWeightedSumAggregator:
    def test_single_task_total(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"a": _loss(2.0)}, {"a": 1.0})
        assert result.total.item() == pytest.approx(2.0)

    def test_weighted_sum(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"a": _loss(2.0), "b": _loss(4.0)}, {"a": 0.5, "b": 0.5})
        assert result.total.item() == pytest.approx(3.0)

    def test_components_namespaced(self) -> None:
        agg = WeightedSumAggregator()
        result = agg.combine({"task": _loss(1.0, "cross_entropy")}, {"task": 1.0})
        assert "task/cross_entropy" in result.components


class TestCriterionParametersAreTrainable:
    """A criterion with learnable parameters must be optimized, checkpointed, and moved to device.

    Metric-learning losses (InfoNCE/SigLIP) hold learnable ``logit_scale``/``bias``; because the
    optimizer is built from ``self.model`` and the criteria live only on the plain ``self.tasks``
    list, those parameters were silently frozen and absent from checkpoints.
    """

    @staticmethod
    def _task_with_parametric_criterion() -> tuple[Task, nn.Parameter]:
        criterion = _ScaledCrossEntropy()
        task = classification("label", num_classes=3)
        return dataclasses.replace(task, criterion=criterion), criterion.logit_scale

    def _lit(self, task: Task) -> LitModule:
        from src.models.backbones import EmbeddingBackbone

        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
        return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))

    def test_criterion_parameter_is_in_the_optimizer(self) -> None:
        task, scale = self._task_with_parametric_criterion()
        optimizer = cast("torch.optim.Optimizer", self._lit(task).configure_optimizers())
        optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
        assert id(scale) in optimized_ids

    def test_criterion_parameter_is_a_registered_submodule(self) -> None:
        task, scale = self._task_with_parametric_criterion()
        lit = self._lit(task)
        assert id(scale) in {id(param) for param in lit.parameters()}  # → device move + checkpoint

    def test_criterion_parameter_is_saved_in_the_state_dict(self) -> None:
        task, _ = self._task_with_parametric_criterion()
        lit = self._lit(task)
        assert any(key.endswith("logit_scale") for key in lit.state_dict())


class TestStepOutputContract:
    """The step returns StepOutput fields via on_*_batch_end."""

    def _lit(self) -> LitModule:
        from src.models.assembly import build_composite_model
        from src.models.backbones import EmbeddingBackbone
        from src.tasks import classification

        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
        return LitModule(model=model, tasks=[task], optimizer_builder=OptimizerBuilder(base_lr=1e-3))

    def _batch(self) -> Any:
        import torch

        from src.core.entities import Batch
        from src.core.keys import IMAGE

        return Batch(inputs={IMAGE: torch.randn(4, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})

    def test_shared_step_returns_loss_and_task_views(self) -> None:
        import torch

        from src.core.entities import TaskStepView, is_step_output

        result = self._lit()._shared_step(self._batch(), Stage.TRAIN)
        assert is_step_output(result)
        assert isinstance(result["loss"], torch.Tensor) and result["loss"].ndim == 0
        assert "output" not in result  # vestigial raw ModelOutput dropped — task_views covers viz
        view = result["task_views"]["label"]
        assert isinstance(view, TaskStepView)
        assert view.predictions.shape == (4, 3)
        assert view.metric_target.shape == (4,)

    def test_task_view_preds_are_detached(self) -> None:
        """predictions feed metrics + visualization only (never backprop), so they carry no graph."""
        result = self._lit()._shared_step(self._batch(), Stage.TRAIN)
        assert result["task_views"]["label"].predictions.requires_grad is False

    def test_validation_step_returns_task_views_for_callbacks(self) -> None:
        """validation_step must return outputs (was None) so on_validation_batch_end gets them."""
        result = self._lit().validation_step(self._batch(), 0)
        assert result is not None
        assert "label" in result["task_views"]


class TestTestVerbose:
    """``trainer.test`` should print Lightning's own table only when our bar is absent."""

    def test_suppressed_when_metrics_progress_bar_present(self) -> None:
        from types import SimpleNamespace

        from src.callbacks.progress_bar import MetricsProgressBar
        from src.composition.wiring.training import _lightning_prints_test_results

        trainer: Any = SimpleNamespace(callbacks=[MetricsProgressBar()])
        assert _lightning_prints_test_results(trainer) is False

    def test_kept_without_metrics_progress_bar(self) -> None:
        from types import SimpleNamespace

        from src.composition.wiring.training import _lightning_prints_test_results

        trainer: Any = SimpleNamespace(callbacks=[object()])
        assert _lightning_prints_test_results(trainer) is True
