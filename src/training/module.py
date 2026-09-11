"""The one Lightning module every run trains through: it keeps the books, the learner does the work.

What a step *is* belongs to the learner; what happens around it belongs here — which stage a value was
produced in, which metrics accumulate it, when they are read out and where the reading goes. That split
is why a second algorithm needs no module of its own, and why this file knows no task kind, no loss and
no backend by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import TYPE_CHECKING, override

import lightning as L
from torch import Tensor, nn

from src.core import Batch, LossOutput, Stage, StepOutput, TensorTree
from src.tracking import MetricKey, report
from src.training.base import FitProfile, Learner, OptimizerFactory, SchedulerFactory

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
    from torch.optim import Optimizer

    from src.metrics import MetricCollection

LOSS = "loss"
"""What the objective is called in a report — the name `scheduler.monitor: val/loss` is written against."""


class TrainingModule(L.LightningModule):
    """Runs a learner through Lightning: one class for every algorithm and every task.

    Metrics arrive as one collection per task and are kept per stage, each with state of its own, so a
    validation epoch never reads rows a training epoch accumulated. Values are read out at the end of
    the epoch they were produced in and routed by what they are: numbers to the log, pictures to the
    trackers that draw pictures.

    The learner sits under ``learner`` and its network under ``learner.model``, which is what a
    checkpoint's keys and a callback's dot-path are written against.
    """

    def __init__(
        self,
        learner: Learner,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None = None,
        metrics: Mapping[str, MetricCollection] | None = None,
    ) -> None:
        super().__init__()
        self.learner = learner
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        declared = dict(metrics or {})
        self._refuse_metrics_for_a_task_that_is_not_learned(declared)
        self._metrics = {stage: {name: collection.clone() for name, collection in declared.items()} for stage in Stage}
        # Registered so Lightning moves them with the model between devices; the mapping above is what
        # reads them back, because torch's containers answer as plain modules. Their state is not
        # checkpointed and needs not be — measured, torchmetrics registers it non-persistent, and every
        # reading below is reset at the end of the epoch that produced it.
        self._measured = nn.ModuleList(collection for stage in self._metrics.values() for collection in stage.values())

    @override
    def training_step(self, batch: Batch, batch_index: int) -> Tensor | None:
        return self._step(batch, Stage.TRAIN)

    @override
    def validation_step(self, batch: Batch, batch_index: int) -> Tensor | None:
        return self._step(batch, Stage.VAL)

    @override
    def test_step(self, batch: Batch, batch_index: int) -> Tensor | None:
        return self._step(batch, Stage.TEST)

    @override
    def on_train_epoch_end(self) -> None:
        self._report(Stage.TRAIN)

    @override
    def on_validation_epoch_end(self) -> None:
        self._report(Stage.VAL)

    @override
    def on_test_epoch_end(self) -> None:
        self._report(Stage.TEST)

    @override
    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        """Build the optimizer over the learner's groups, and the schedule once the fit has a length."""
        optimizer = self._optimizer_factory(self.learner.parameter_groups())
        if self._scheduler_factory is None:
            return optimizer
        optimized: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": self._scheduler_factory(optimizer, self._profile()),
        }
        return optimized

    def _profile(self) -> FitProfile:
        """How long this fit turned out to be — the one question only the trainer can answer.

        Read from ``estimated_stepping_batches`` because it is what Lightning guarantees at this point:
        the loops are not set up yet, so the per-epoch counts are still infinite.
        """
        steps, epochs = self.trainer.estimated_stepping_batches, self.trainer.max_epochs
        # Measured on lightning 2.6.5: a loop with no length reports -1 rather than infinity.
        if not isfinite(steps) or steps < 1 or epochs is None or epochs < 1:
            raise ValueError(
                f"A schedule is sized by the length of the fit, and this one has no declared end: {steps} "
                f"optimizer steps over {epochs} epochs. Declare epochs, or run without a scheduler."
            )
        return FitProfile(total_steps=int(steps), epochs=epochs)

    def _step(self, batch: Batch, stage: Stage) -> Tensor | None:
        output = self.learner.step(batch)
        if output.loss is None and stage is Stage.TRAIN:
            raise ValueError(
                f"{type(self.learner).__name__} produced no loss on a training step, so there is nothing to "
                "descend. A learner may leave the loss out in evaluation alone."
            )
        for name, collection in self._metrics[stage].items():
            predicted, wanted = self._scored(output, name, stage)
            collection.update(predicted, wanted)
        if output.loss is None:
            return None
        self._log_objective(output.loss, stage, len(batch))
        return output.loss.total

    def _scored(self, output: StepOutput, task: str, stage: Stage) -> tuple[TensorTree, TensorTree]:
        """What one task's metrics read, refused by name when the step answered for something else.

        A metric fed nothing would report a score for a task the run never actually answered, which is
        the silent failure this replaces: the parts are assembled separately and have to agree.
        """
        if task not in output.predictions or task not in output.targets:
            answered = ", ".join(sorted(output.predictions.keys() & output.targets.keys())) or "nothing"
            raise ValueError(
                f"Task {task!r} is measured in {stage}, but this step answered for {answered}: its metrics "
                "would be computed over no rows at all."
            )
        return output.predictions[task], output.targets[task]

    def _log_objective(self, loss: LossOutput, stage: Stage, count: int) -> None:
        """The total, then every term it is made of; a term already carries its task in its own name.

        Epoch-level throughout: a per-step loss is noise at the resolution a chart is read at, and the
        epoch value is the run's rather than rank zero's. Only the total reaches the progress bar, and
        only while training — the terms are what a report is for.
        """
        self.log(
            str(MetricKey(stage, LOSS)),
            loss.total,
            prog_bar=stage is Stage.TRAIN,
            batch_size=count,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        for name, value in loss.breakdown().items():
            self.log(str(MetricKey(stage, name)), value, batch_size=count, on_step=False, on_epoch=True, sync_dist=True)

    def _report(self, stage: Stage) -> None:
        """Read this stage's metrics out, send each value where its shape belongs, and start over.

        A sanity check is not an epoch: Lightning suppresses ``self.log`` during one, but nothing
        suppresses handing a picture straight to a tracker, and an untrained network would be drawn
        at the same iteration as the first real epoch. Its readings are dropped rather than reported.
        """
        for name, collection in self._metrics[stage].items():
            if self._trainer is not None and self.trainer.sanity_checking:
                collection.reset()
                continue
            classes = self.learner.tasks[name].info.classes
            for label, value in collection.compute().items():
                report(
                    MetricKey(stage, label, task=name),
                    value,
                    scalar_log=self.log,
                    trackers=self.loggers,
                    step=self.current_epoch,
                    classes=classes,
                )
            collection.reset()

    def _refuse_metrics_for_a_task_that_is_not_learned(self, metrics: Mapping[str, object]) -> None:
        unknown = sorted(metrics.keys() - self.learner.tasks.keys())
        if unknown:
            raise ValueError(
                f"Metrics are declared for {', '.join(unknown)}, which this run does not learn: it has "
                f"{', '.join(sorted(self.learner.tasks)) or 'no tasks'}. Nothing would ever update them."
            )
