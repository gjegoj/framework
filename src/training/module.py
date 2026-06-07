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

from typing import Any

import lightning as L
import torch

from src.core.entities import Batch, LossResult, Task
from src.core.enums import Stage
from src.core.ports import LossAggregator
from src.models.assembly import CompositeModel
from src.training.aggregator import WeightedSumAggregator
from src.training.optimizer import OptimizerBuilder


class LitModule(L.LightningModule):
    """Lightning training/evaluation module for multi-task vision models.

    Parameters:
        model (CompositeModel): Shared backbone + heads.
        tasks (list[Task]): Task bundles (codec/criterion/activation/metrics).
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
        aggregator (LossAggregator | None): Loss combiner; defaults to weighted sum.
        task_lr_overrides (dict[str, float] | None): Per-head LR for the optimizer.
    """

    def __init__(
        self,
        model: CompositeModel,
        tasks: list[Task],
        optimizer_builder: OptimizerBuilder,
        aggregator: LossAggregator | None = None,
        task_lr_overrides: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tasks = tasks
        self._task_map: dict[str, Task] = {t.name: t for t in tasks}
        self._optimizer_builder = optimizer_builder
        self._aggregator: LossAggregator = aggregator or WeightedSumAggregator()
        self._task_lr_overrides = task_lr_overrides or {}

    # ------------------------------------------------------------------ steps

    def _shared_step(self, batch: Batch | dict[str, Any], stage: Stage) -> torch.Tensor:
        if isinstance(batch, dict):
            batch = Batch(**batch)

        output = self.model(batch.inputs)
        losses: dict[str, LossResult] = {}

        for task in self.tasks:
            logits = output.task_logits[task.name]
            target = task.codec.adapt(batch.targets[task.name])
            losses[task.name] = task.criterion(logits, target.loss)
            task.metrics[stage].update(task.activation(logits), target.metric)

        weights = {t.name: t.weight for t in self.tasks}
        combined = self._aggregator.combine(losses, weights)
        self._log_losses(combined, stage)
        return combined.total

    def training_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, Stage.TRAIN)

    def validation_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, Stage.VAL)

    def test_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> None:
        self._shared_step(batch, Stage.TEST)

    # --------------------------------------------------------- device hooks

    def _move_metrics_to_device(self) -> None:
        for task in self.tasks:
            for metric_set in task.metrics.values():
                metric_set.to(self.device)

    def on_fit_start(self) -> None:
        self._move_metrics_to_device()

    def on_test_start(self) -> None:
        self._move_metrics_to_device()

    # ----------------------------------------------------------- epoch hooks

    def _shared_epoch_end(self, stage: Stage) -> None:
        for task in self.tasks:
            metrics = task.metrics[stage].compute()
            for metric_name, value in metrics.items():
                self._log_metric(f"{task.name}/{metric_name}/{stage}", value)
            task.metrics[stage].reset()

    def _log_metric(self, key: str, value: Any) -> None:
        """Log a metric value, handling per-class tensors gracefully.

        - Scalar (0-D tensor or number) → logged directly.
        - 1-D tensor (per-class, ``average="none"``) → mean logged at ``key``;
          per-class values at ``{key}_class{i}`` (no prog_bar).
        - 2-D+ (e.g. confusion matrix) → skipped until typed metrics (M4).
        """
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            self.log(key, value, prog_bar=True)
            return
        if value.ndim == 1:
            self.log(key, value.float().mean(), prog_bar=True)
            for i, v in enumerate(value):
                self.log(f"{key}_class{i}", v.float(), prog_bar=False)
        # 2-D+ tensors (confusion matrix, etc.) are skipped here; M4 adds typed logging.

    def on_train_epoch_start(self) -> None:
        for task in self.tasks:
            task.metrics[Stage.TRAIN].reset()

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TRAIN)

    def on_validation_epoch_start(self) -> None:
        for task in self.tasks:
            task.metrics[Stage.VAL].reset()

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.VAL)

    def on_test_epoch_start(self) -> None:
        for task in self.tasks:
            task.metrics[Stage.TEST].reset()

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TEST)

    # ----------------------------------------------------------- optimizer

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self._optimizer_builder.build(self.model, task_lr_overrides=self._task_lr_overrides)

    # ---------------------------------------------------------------- utils

    def _log_losses(self, combined: LossResult, stage: Stage) -> None:
        self.log(
            f"loss/{stage}/total",
            combined.total,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        for name, value in combined.components.items():
            self.log(f"loss/{stage}/{name}", value, on_step=False, on_epoch=True)
