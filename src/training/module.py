"""LitModule: the humble Lightning training module.

Orchestrates one training step: forward → per-task loss → aggregation. All
decision logic lives in the Task objects and the aggregator; LitModule is a
thin coordinator that also handles logging and optimizer configuration.

The step loop from the plan:

    features = model.backbone(batch.inputs)
    for task in tasks:
        logits = model.heads[task.name](features[task.feature_key])
        target = task.codec.adapt(batch.targets[task.name])
        losses[task.name] = task.criterion(logits, target.loss)
        task.metrics[stage].update(task.activation(logits), target.metric)
    total = aggregator.combine(losses, weights)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightning as L
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler, OptimizerLRSchedulerConfig

from src.core.entities import Batch, LossResult, StepOutput, Task, TaskStepView
from src.core.enums import Stage
from src.core.ports import LossAggregator
from src.metrics.directions import task_metric_directions
from src.metrics.handlers import (
    DEFAULT_METRIC_HANDLERS,
    MetricHandler,
    MetricLogContext,
    dispatch,
)
from src.models.assembly import CompositeModel
from src.training.aggregator import WeightedSumAggregator
from src.training.optimizer import OptimizerBuilder
from src.training.scheduler import TRAINER_FACTS, SchedulerBuilder


class LitModule(L.LightningModule):
    """Lightning training/evaluation module for multi-task vision models.

    Parameters:
        model (CompositeModel): Shared backbone + heads.
        tasks (list[Task]): Task bundles (codec/criterion/activation/metrics).
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
        scheduler_builder (SchedulerBuilder | None): Optional LR scheduler builder.
        aggregator (LossAggregator | None): Loss combiner; defaults to weighted sum.
        task_lr_overrides (dict[str, float] | None): Per-head LR for the optimizer.
    """

    def __init__(
        self,
        model: CompositeModel,
        tasks: list[Task],
        optimizer_builder: OptimizerBuilder,
        scheduler_builder: SchedulerBuilder | None = None,
        aggregator: LossAggregator | None = None,
        task_lr_overrides: dict[str, float] | None = None,
        metric_handlers: Sequence[MetricHandler] = DEFAULT_METRIC_HANDLERS,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tasks = tasks
        self._task_map: dict[str, Task] = {t.name: t for t in tasks}
        self._optimizer_builder = optimizer_builder
        self._scheduler_builder = scheduler_builder
        self._aggregator: LossAggregator = aggregator or WeightedSumAggregator()
        self._task_lr_overrides = task_lr_overrides or {}
        self._metric_handlers: tuple[MetricHandler, ...] = tuple(metric_handlers)
        self._hparams_to_log = hparams

    # ------------------------------------------------------------------ steps

    def _shared_step(self, batch: Batch | dict[str, Any], stage: Stage) -> StepOutput:
        """Run the forward + loss/metric loop and return step artifacts.

        Returns a :class:`StepOutput` dict. ``task_views`` (post-activation
        preds + metric targets) flow to ``on_*_batch_end(outputs, ...)`` so
        visualization callbacks reuse step work without re-running activation or
        codec adaptation. ``preds`` are detached — the activation output only
        feeds metrics/inference, never backprop (the loss runs on ``logits``).
        """
        if isinstance(batch, dict):
            batch = Batch(**batch)

        output = self.model(batch.inputs)
        losses: dict[str, LossResult] = {}
        task_views: dict[str, TaskStepView] = {}

        for task in self.tasks:
            logits = output.task_logits[task.name]
            target = task.codec.adapt(batch.targets[task.name])
            preds = task.activation(logits).detach()
            losses[task.name] = task.criterion(logits, target.loss)
            task.metrics[stage].update(preds, target.metric)
            task_views[task.name] = TaskStepView(preds=preds, metric_target=target.metric)

        weights = {t.name: t.weight for t in self.tasks}
        combined = self._aggregator.combine(losses, weights)
        self._log_losses(combined, stage)
        return {"loss": combined.total, "task_views": task_views}

    def training_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._shared_step(batch, Stage.TRAIN)

    def validation_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._shared_step(batch, Stage.VAL)

    def test_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._shared_step(batch, Stage.TEST)

    # --------------------------------------------------------- device hooks

    def _move_metrics_to_device(self) -> None:
        for task in self.tasks:
            for metric_set in task.metrics.values():
                metric_set.to(self.device)

    def on_fit_start(self) -> None:
        self._move_metrics_to_device()
        if self._hparams_to_log is not None and isinstance(self.logger, LightningLogger) and self.global_rank == 0:
            self.logger.log_hyperparams(self._hparams_to_log)

    def on_test_start(self) -> None:
        self._move_metrics_to_device()

    # ----------------------------------------------------------- epoch hooks

    def _shared_epoch_end(self, stage: Stage) -> None:
        for task in self.tasks:
            metrics = task.metrics[stage].compute()
            for metric_name, value in metrics.items():
                ctx = MetricLogContext(
                    log_scalar=lambda key, val: self.log(key, val, prog_bar=True),
                    logger=self.logger,
                    step=self.current_epoch,
                    class_names=task.class_names,
                    metric_name=metric_name,
                )
                dispatch(f"{task.name}/{metric_name}/{stage}", value, ctx, self._metric_handlers)
            task.metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TEST)

    # ----------------------------------------------------------- optimizer

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._optimizer_builder.build(self.model, task_lr_overrides=self._task_lr_overrides)
        if self._scheduler_builder is None:
            return optimizer
        trainer_facts = {name: getattr(self.trainer, attr) for name, attr in TRAINER_FACTS.items()}
        config: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": self._scheduler_builder.build(optimizer, trainer_facts),
        }
        return config

    # ---------------------------------------------------------------- utils

    def metric_directions(self) -> dict[str, bool | None]:
        """Per-metric ``higher_is_better`` keyed as this module logs (``MetricDirectionProvider``)."""
        return task_metric_directions(self.tasks)

    def _log_losses(self, combined: LossResult, stage: Stage) -> None:
        self.log(
            f"loss/{stage}/total",
            combined.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        for name, value in combined.components.items():
            self.log(f"loss/{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
