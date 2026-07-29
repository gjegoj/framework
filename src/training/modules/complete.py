"""Complete-model regime: the framework's Lightning contour driving a fused model.

The generic step shell for any ``CompleteModel`` family: native batches bypass the
``Batch`` normalization, the model computes its own loss, and evaluation feeds a
``MetricBundle``. Optimizer configuration, loss-log grammar, checkpoint pruning and
hparams are inherited from ``BaseLitModule``.
"""

from __future__ import annotations

from typing import Any

import torch

from src.core.entities import Batch, LossResult, StepOutput
from src.core.enums import Stage
from src.metrics.bundle import MetricBundle
from src.models.complete import CompleteModel
from src.training.modules.base import BaseLitModule
from src.training.optim.optimizer import OptimizerBuilder
from src.training.optim.scheduler import SchedulerBuilder

_EVALUATION_STAGES = (Stage.VAL, Stage.TEST)


class CompleteModelLitModule(BaseLitModule[CompleteModel[Any, Any]]):
    """Drives a complete model: its loss on TRAIN, decoded predictions + bundle on VAL/TEST.

    Losses log through the standard contour (``loss/<stage>/total`` plus
    ``<task>/<component>``); bundle leaves log at epoch end as
    ``<task>/<leaf>/<stage>`` — the framework's ``task/metric/stage`` grammar, so
    the progress bar and ClearML parse them as usual.

    Parameters:
        model (CompleteModel[Any, Any]): The fused model (e.g. ``YoloModel``).
        task_name (str): Task name prefixed to every logged key.
        metric_bundle (MetricBundle[Any, Any]): Evaluation metrics paired with the model.
        optimizer_builder (OptimizerBuilder): Builds the optimizer on configure.
        scheduler_builder (SchedulerBuilder | None): Optional LR scheduler builder.
        hparams (dict[str, Any] | None): Config snapshot logged at ``on_fit_start``.
    """

    def __init__(
        self,
        model: CompleteModel[Any, Any],
        task_name: str,
        metric_bundle: MetricBundle[Any, Any],
        optimizer_builder: OptimizerBuilder,
        scheduler_builder: SchedulerBuilder | None = None,
        hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            tasks=[],
            optimizer_builder=optimizer_builder,
            scheduler_builder=scheduler_builder,
            hparams=hparams,
        )
        self._task_name = task_name
        self._metric_bundle = metric_bundle

    # ------------------------------------------------------------------ steps

    def _shared_step(self, batch: Any, stage: Stage) -> StepOutput:
        """Not applicable: complete models override the step methods directly.

        The base's ``_shared_step`` seam assumes the framework ``Batch`` entity;
        complete-model batches are their own native dicts, so this regime replaces
        ``training_step``/``validation_step``/``test_step`` instead. Reaching this
        method means a subclass wired the base dispatch back in by mistake.
        """
        raise NotImplementedError("CompleteModelLitModule overrides the step methods; _shared_step does not apply.")

    def training_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        prepared = self._prepare_batch(batch)
        loss = self.model.training_loss(prepared)
        self._log_namespaced(loss, Stage.TRAIN)
        return {"loss": loss.total, "task_views": {}}

    def validation_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._evaluation_step(self._prepare_batch(batch), Stage.VAL)

    def test_step(self, batch: Batch | dict[str, Any], batch_idx: int) -> StepOutput:
        return self._evaluation_step(self._prepare_batch(batch), Stage.TEST)

    def _prepare_batch(self, batch: Batch | dict[str, Any]) -> dict[str, Any]:
        """Narrow to the native dict and apply the model's batch normalization."""
        if not isinstance(batch, dict):
            raise TypeError(f"CompleteModelLitModule expects a native batch dict, got {type(batch).__name__}.")
        return self.model.prepare_batch(batch)

    def _evaluation_step(self, batch: dict[str, Any], stage: Stage) -> StepOutput:
        """One forward pass feeds both the loss and the metric bundle."""
        with torch.no_grad():
            output = self.model(batch)
        loss = self.model.evaluation_loss(batch, output)
        self._log_namespaced(loss, stage)
        self._metric_bundle.update(stage, self.model.predictions(output), self.model.targets(batch))
        return {"loss": loss.total, "task_views": {}}

    def _log_namespaced(self, loss: LossResult, stage: Stage) -> None:
        namespaced = {f"{self._task_name}/{name}": value for name, value in loss.components.items()}
        self._log_losses(LossResult(total=loss.total, components=namespaced), stage)

    # ------------------------------------------------------------ epoch hooks

    def on_validation_epoch_end(self) -> None:
        self._report_bundle(Stage.VAL)

    def on_test_epoch_end(self) -> None:
        self._report_bundle(Stage.TEST)

    def _report_bundle(self, stage: Stage) -> None:
        for leaf, value in self._metric_bundle.log_items(stage).items():
            self.log(f"{self._task_name}/{leaf}/{stage}", value, prog_bar=True)
        self._metric_bundle.reset(stage)

    # ---------------------------------------------------------------- utils

    def metric_directions(self) -> dict[str, bool | None]:
        """Bundle directions namespaced per evaluation stage for the progress bar."""
        return {
            f"{self._task_name}/{leaf}/{stage}": direction
            for leaf, direction in self._metric_bundle.directions().items()
            for stage in _EVALUATION_STAGES
        }
