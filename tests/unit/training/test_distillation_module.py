"""TeacherEnsemble and DistillationLitModule: averaging, freezing, loss composition."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.core.entities import Batch, FeatureBundle, LossResult, ModelOutput
from src.core.enums import Stage
from src.losses.distillation import KLDivergenceCriterion
from src.models import build_composite_model
from src.models.backbones import EmbeddingBackbone
from src.models.ensemble import TeacherEnsemble
from src.tasks import classification
from src.training.modules.distillation import DistillationLitModule
from src.training.optim import OptimizerBuilder


class _FixedLogitsModel(nn.Module):
    """Stands in for a teacher CompositeModel: returns constant logits per task."""

    def __init__(self, logits: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self._logits = logits
        self.linear = nn.Linear(2, 2)  # gives the "model" parameters to freeze

    def forward(self, inputs: dict[str, torch.Tensor]) -> ModelOutput:
        return ModelOutput(features=FeatureBundle(), task_logits=dict(self._logits))


class TestTeacherEnsemble:
    def test_raw_logit_mean_across_teachers(self) -> None:
        first = _FixedLogitsModel({"label": torch.zeros(2, 3)})
        second = _FixedLogitsModel({"label": torch.full((2, 3), 2.0)})
        ensemble = TeacherEnsemble([first, second])
        averaged = ensemble({"image": torch.randn(2, 4)})
        assert torch.allclose(averaged["label"], torch.ones(2, 3))

    def test_teachers_frozen_and_output_detached(self) -> None:
        ensemble = TeacherEnsemble([_FixedLogitsModel({"label": torch.randn(2, 3)})])
        assert all(not parameter.requires_grad for parameter in ensemble.parameters())
        assert not ensemble.training
        assert not ensemble({"image": torch.randn(2, 4)})["label"].requires_grad

    def test_empty_teachers_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one teacher"):
            TeacherEnsemble([])


class TestDistillationLitModule:
    @staticmethod
    def _module(distillation_weight: float = 0.5) -> DistillationLitModule:
        """Real student (embedding backbone + classification head) distilled by one fixed teacher."""
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"label": task.head_spec})
        ensemble = TeacherEnsemble([_FixedLogitsModel({"label": torch.randn(4, 3)})])
        return DistillationLitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            teachers=ensemble,
            distillation_criteria={"label": KLDivergenceCriterion(temperature=2.0)},
            distillation_weights={"label": distillation_weight},
        )

    @staticmethod
    def _batch() -> Batch:
        return Batch(inputs={"image": torch.randn(4, 16)}, targets={"label": torch.tensor([0, 1, 2, 0])})

    def _combined_loss(self, module: DistillationLitModule, stage: Stage) -> LossResult:
        """Capture the LossResult passed to _log_losses without changing production code.

        The aggregator namespaces components as ``<task>/<component>``, so the KL term
        surfaces as ``label/kl``.
        """
        captured: list[LossResult] = []

        def capture(combined_loss: LossResult, stage: Stage) -> None:
            captured.append(combined_loss)

        module._log_losses = capture  # type: ignore[method-assign]
        module._shared_step(self._batch(), stage)
        return captured[0]

    def test_train_step_adds_kl_component(self) -> None:
        combined = self._combined_loss(self._module(), Stage.TRAIN)
        assert "label/kl" in combined.components
        assert torch.isfinite(combined.total)

    def test_validation_step_has_no_kl(self) -> None:
        combined = self._combined_loss(self._module(), Stage.VAL)
        assert "label/kl" not in combined.components

    def test_teachers_absent_from_state_dict(self) -> None:
        keys = self._module().state_dict().keys()
        assert not any("teacher" in key.lower() for key in keys)

    def test_unknown_distillation_task_rejected(self) -> None:
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"label": task.head_spec})
        ensemble = TeacherEnsemble([_FixedLogitsModel({"label": torch.randn(4, 3)})])
        with pytest.raises(ValueError, match="unknown task"):
            DistillationLitModule(
                model=model,
                tasks=[task],
                optimizer_builder=OptimizerBuilder(base_lr=1e-3),
                teachers=ensemble,
                distillation_criteria={"missing": KLDivergenceCriterion()},
                distillation_weights={"missing": 0.5},
            )

    def test_missing_weight_rejected_at_construction(self) -> None:
        """A criteria/weights key mismatch must fail at build time, not on the first train batch."""
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"label": task.head_spec})
        ensemble = TeacherEnsemble([_FixedLogitsModel({"label": torch.randn(4, 3)})])
        with pytest.raises(ValueError, match="missing entries"):
            DistillationLitModule(
                model=model,
                tasks=[task],
                optimizer_builder=OptimizerBuilder(base_lr=1e-3),
                teachers=ensemble,
                distillation_criteria={"label": KLDivergenceCriterion()},
                distillation_weights={},
            )

    def test_teachers_skipped_when_no_task_distills(self) -> None:
        """With empty distillation criteria the teacher forward must not run at all."""
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"label": task.head_spec})

        class _ExplodingTeacher(_FixedLogitsModel):
            def forward(self, inputs: dict[str, torch.Tensor]) -> ModelOutput:
                raise AssertionError("teacher forward must be short-circuited")

        module = DistillationLitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            teachers=TeacherEnsemble([_ExplodingTeacher({"label": torch.randn(4, 3)})]),
            distillation_criteria={},
            distillation_weights={},
        )
        output = module._shared_step(self._batch(), Stage.TRAIN)
        assert torch.isfinite(output["loss"])


class _ParametricDistillationCriterion(KLDivergenceCriterion):
    """KL plus a learnable scale — models a parametric distillation loss (e.g. learnable T)."""

    def __init__(self) -> None:
        super().__init__(temperature=2.0)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> LossResult:
        base = super().forward(logits, target)
        return LossResult(total=self.scale * base.total, components=base.components)


class TestParametricDistillationCriterion:
    """A distillation loss with learnable parameters must be a first-class module citizen."""

    @staticmethod
    def _module() -> DistillationLitModule:
        task = classification("label", num_classes=3)
        model = build_composite_model(EmbeddingBackbone(embedding_dim=16), {"label": task.head_spec})
        return DistillationLitModule(
            model=model,
            tasks=[task],
            optimizer_builder=OptimizerBuilder(base_lr=1e-3),
            teachers=TeacherEnsemble([_FixedLogitsModel({"label": torch.randn(4, 3)})]),
            distillation_criteria={"label": _ParametricDistillationCriterion()},
            distillation_weights={"label": 0.5},
        )

    def test_parameters_reach_the_optimizer(self) -> None:
        module = self._module()
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        optimizer_parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        scale = module._distillation_criteria["label"].scale
        assert id(scale) in optimizer_parameter_ids

    def test_parameters_saved_in_state_dict(self) -> None:
        assert "_distillation_criteria.label.scale" in self._module().state_dict()
