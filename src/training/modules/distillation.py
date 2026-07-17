"""Knowledge distillation regime: LitModule extended with online teacher soft-targets."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Any, cast

import torch.nn as nn
from torch import Tensor

from src.core.entities import Batch, LossResult, TargetView, Task
from src.core.enums import Stage
from src.core.ports import Criterion, LossAggregator
from src.metrics.reporter import MetricReporter
from src.models.assembly import CompositeModel
from src.models.ensemble import TeacherEnsemble
from src.training.modules.lit_module import LitModule
from src.training.optim.optimizer import OptimizerBuilder
from src.training.optim.scheduler import SchedulerBuilder


class DistillationLitModule(LitModule):
    """Standard supervised training plus additive online distillation.

    Distillation is the standard step with one extra signal, so this extends
    :class:`LitModule` and overrides only its two seams instead of re-implementing the
    step loop: :meth:`_auxiliary_targets` supplies the teachers' averaged soft targets
    (TRAIN only), and :meth:`_task_loss` forms ``loss = hard + weight * soft`` per task,
    where ``hard`` is the task's own criterion on ground truth. Because soft targets are
    empty off TRAIN, validation/test losses stay pure task losses — checkpoint monitoring
    is comparable with non-distilled runs and teachers never run during evaluation.

    The teacher ensemble is deliberately kept OUT of the module tree (a plain-list holder)
    so it stays invisible to ``state_dict``, checkpoints and EMA weight averaging. Lightning
    therefore never moves it, so :meth:`on_fit_start` places it on the training device/dtype
    (after Lightning's own transfer). Distillation criteria, by contrast, ARE registered
    (an ``nn.ModuleDict``) so a parametric distillation loss is device-moved, checkpointed,
    and — via :meth:`_extra_optimizer_parameters` — trained.

    Parameters:
        model (CompositeModel): Student backbone + heads.
        tasks (list[Task]): Task bundles (adapter/criterion/activation/metrics).
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
        teachers (TeacherEnsemble): Frozen soft-target providers.
        distillation_criteria (dict[str, Criterion]): Soft-loss brick per distilled task.
        distillation_weights (dict[str, float]): Additive soft-loss weight per distilled task.
        scheduler_builder (SchedulerBuilder | None): Optional LR scheduler builder.
        loss_aggregator (LossAggregator | None): Loss combiner; defaults to weighted sum.
        metric_reporter (MetricReporter | None): Logs metrics by shape; defaults to the standard chain.
        hparams (dict[str, Any] | None): Config snapshot logged at ``on_fit_start``.
    """

    def __init__(
        self,
        model: CompositeModel,
        tasks: list[Task],
        optimizer_builder: OptimizerBuilder,
        teachers: TeacherEnsemble,
        distillation_criteria: dict[str, Criterion],
        distillation_weights: dict[str, float],
        scheduler_builder: SchedulerBuilder | None = None,
        loss_aggregator: LossAggregator | None = None,
        metric_reporter: MetricReporter | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            tasks=tasks,
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            loss_aggregator=loss_aggregator,
            metric_reporter=metric_reporter,
            hparams=hparams,
        )
        task_names = {task.name for task in tasks}
        unknown = set(distillation_criteria) - task_names
        if unknown:
            raise ValueError(f"distillation_criteria reference unknown task(s): {sorted(unknown)}.")
        missing_weights = set(distillation_criteria) - set(distillation_weights)
        if missing_weights:
            raise ValueError(f"distillation_weights is missing entries for task(s): {sorted(missing_weights)}.")
        # Plain-list holder — assigning an nn.Module attribute would register the teachers
        # as a submodule and leak them into state_dict / EMA averaging / checkpoints.
        self._teacher_holder = [teachers]
        # ModuleDict (unlike the teachers) so a parametric distillation loss is device-moved,
        # checkpointed, and trained — mirroring how the base registers task criteria.
        self._distillation_criteria = nn.ModuleDict(distillation_criteria)
        self._distillation_weights = distillation_weights

    def on_fit_start(self) -> None:
        super().on_fit_start()  # base moves metrics + logs hparams, AFTER Lightning's device transfer
        # Teachers are off the module tree, so Lightning's device/dtype transfer never reaches
        # them; align them to the student's own parameter device and dtype (fp32/half) so the
        # soft-target forward matches the student even under true-half precision.
        reference = next(self.model.parameters(), None)
        if reference is not None:
            self._teacher_holder[0].to(device=reference.device, dtype=reference.dtype)

    def _extra_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        return itertools.chain(super()._extra_optimizer_parameters(), self._distillation_criteria.parameters())

    def _auxiliary_targets(self, batch: Batch, stage: Stage) -> dict[str, Tensor]:
        # Short-circuit the teacher forward off TRAIN or when no task distills.
        if stage is Stage.TRAIN and len(self._distillation_criteria) > 0:
            return self._teacher_holder[0](batch.inputs)
        return {}

    def _task_loss(self, task: Task, logits: Tensor, target: TargetView, auxiliary: dict[str, Tensor]) -> LossResult:
        hard = task.criterion(logits, target.loss)
        if task.name not in auxiliary or task.name not in self._distillation_criteria:
            return hard
        criterion = cast("Criterion", self._distillation_criteria[task.name])
        soft = criterion(logits, auxiliary[task.name])
        weight = self._distillation_weights[task.name]
        return LossResult(total=hard.total + weight * soft.total, components={**hard.components, **soft.components})
