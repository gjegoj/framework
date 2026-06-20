"""BaseLitModule: the shared Lightning scaffolding for training/eval modules.

Holds everything common across training regimes — construction, the step dispatch,
device/epoch hooks, optimizer configuration, loss/metric logging — leaving only the
per-batch ``_shared_step`` (forward → loss → aggregate) for subclasses. ``LitModule`` is
the standard supervised implementation; a knowledge-distillation module would subclass this
and override ``_shared_step`` (teacher forward + KD loss) while reusing the rest unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler, OptimizerLRSchedulerConfig

from src.core.entities import Batch, LossResult, StepOutput, Task
from src.core.enums import Stage
from src.core.keys import LOSS, TOTAL
from src.core.ports import LossAggregator
from src.metrics.directions import task_metric_directions
from src.metrics.reporter import MetricReporter
from src.models.assembly import CompositeModel
from src.training.aggregator import WeightedSumAggregator
from src.training.optim.optimizer import OptimizerBuilder
from src.training.optim.scheduler import TRAINER_FACTS, SchedulerBuilder


class BaseLitModule(L.LightningModule, ABC):
    """Shared scaffolding for multi-task Lightning modules; subclasses implement ``_shared_step``.

    Parameters:
        model (CompositeModel): Shared backbone + heads.
        tasks (list[Task]): Task bundles (adapter/criterion/activation/metrics).
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
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
        scheduler_builder: SchedulerBuilder | None = None,
        loss_aggregator: LossAggregator | None = None,
        metric_reporter: MetricReporter | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.tasks = tasks
        self._task_weights: dict[str, float] = {task.name: task.weight for task in tasks}
        self._optimizer_builder = optimizer_builder
        self._scheduler_builder = scheduler_builder
        self._loss_aggregator: LossAggregator = loss_aggregator or WeightedSumAggregator()
        self._metric_reporter = metric_reporter or MetricReporter()
        self._hparams_to_log = hparams

    # ------------------------------------------------------------------ steps

    @abstractmethod
    def _shared_step(self, batch: Batch, stage: Stage) -> StepOutput:
        """Run one forward + loss/metric step on a normalized ``Batch`` (regime-specific)."""

    def _run_step(self, batch: Batch | dict[str, Any], stage: Stage) -> StepOutput:
        """Normalize the Lightning batch (collate may yield a dict) and dispatch to ``_shared_step``."""
        return self._shared_step(Batch(**batch) if isinstance(batch, dict) else batch, stage)

    def training_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._run_step(batch, Stage.TRAIN)

    def validation_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._run_step(batch, Stage.VAL)

    def test_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._run_step(batch, Stage.TEST)

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
            self._metric_reporter.report(task, stage, self._log_metric, self.logger, self.current_epoch)

    def _log_metric(self, key: str, value: Any) -> None:
        """Scalar sink handed to the reporter; logs to the progress bar + logger."""
        self.log(key, value, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TRAIN)

    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TEST)

    # ----------------------------------------------------------- optimizer

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self._optimizer_builder.build(self.model)
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

    def _log_losses(self, combined_loss: LossResult, stage: Stage) -> None:
        self.log(
            f"{LOSS}/{stage}/{TOTAL}",
            combined_loss.total,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        for name, value in combined_loss.components.items():
            self.log(f"{LOSS}/{stage}/{name}", value, on_step=False, on_epoch=True, sync_dist=True)
