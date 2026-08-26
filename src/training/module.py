"""The single Lightning module serving every ``Model`` family."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, override

import lightning as L
from torch import nn

from src.core import log_keys
from src.core.entities import LightningStepOutput, StepPreview
from src.core.ports import AwaitsPreview
from src.core.reporting import report_metric
from src.core.taxonomy import Stage
from src.training.optim import FitProfile

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
    from torch.optim import Optimizer

    from src.core.entities import Batch, Loss, StepResult, Task
    from src.core.ports import Model
    from src.training.optim import OptimizerFactory, SchedulerFactory

SHARED_GROUP: Final = "backbone"
"""What the parameter group holding everything no task claims is called in a chart.

Defined by subtraction — the parameters left once every task has taken its head
and criterion — and named for what that remainder *is*: the encoder, in the
composite family and in the student of a distilled one alike. A label, not an
address: the dot-path a config freezes by is published as ``CompositeModel.BACKBONE``
and the two are free to drift, since renaming the attribute does not rename the
pace a reader is comparing.
"""


class TrainingModule(L.LightningModule):
    """Runs any ``Model`` through Lightning: one class for every family.

    The unified ``Model.step`` contract keeps this module family-agnostic —
    composite, vendor-native, and decorated models all train through the same
    code path. Metric containers from the tasks are registered as submodules
    so Lightning moves them across devices with the model.

    Log keys follow ``{stage}/...``: the total as ``{stage}/loss``, loss parts
    as ``{stage}/{task}/{part}``, metrics as ``{stage}/{task}/{metric}``.
    """

    MODEL: ClassVar[str] = "model"
    """The attribute the model sits under, and so the head of every dot-path into it.

    Published rather than assumed: a freeze callback names the backbone by dot-path
    in config, and a checkpoint's keys are this attribute's own under this name.
    Both readers compose the path from here, so the copies of the string cannot
    drift apart.
    """

    def __init__(
        self,
        model: Model,
        tasks: Sequence[Task],
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None = None,
    ) -> None:
        super().__init__()
        # Public, unlike this module's other state: a config freezing part of a model
        # names it by dot-path, so the attribute is part of the contract.
        self.model = model
        self._tasks = list(tasks)
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._metric_containers = nn.ModuleList(metric_set for task in tasks for metric_set in task.metrics.values())

    @override
    def training_step(self, batch: Batch, batch_index: int) -> LightningStepOutput:
        return self._shared_step(batch, Stage.TRAIN)

    @override
    def validation_step(self, batch: Batch, batch_index: int) -> LightningStepOutput:
        return self._shared_step(batch, Stage.VAL)

    @override
    def test_step(self, batch: Batch, batch_index: int) -> LightningStepOutput:
        return self._shared_step(batch, Stage.TEST)

    @override
    def on_train_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TRAIN)

    @override
    def on_validation_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.VAL)

    @override
    def on_test_epoch_end(self) -> None:
        self._shared_epoch_end(Stage.TEST)

    @override
    def configure_optimizers(self) -> Optimizer | OptimizerLRSchedulerConfig:
        """Build the optimizer, and the scheduler once the trainer can be read."""
        optimizer = self._optimizer_factory(self._parameter_groups())
        if self._scheduler_factory is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": self._scheduler_factory(optimizer, self._fit_profile())}

    def _parameter_groups(self) -> list[dict[str, Any]]:
        """One named group per task with components of its own, and one for what they share.

        Split whether or not a rate was declared, because the groups are what a learning-rate
        monitor draws (measured: AdamW through one group or three moves every parameter the
        same). A group carries an ``lr`` only where its task declared one; the base rate has
        one home, the optimizer section.
        """
        groups: list[dict[str, Any]] = []
        owned: set[int] = set()
        for task in self._tasks:
            parameters = list(self.model.task_parameters(task.name))
            if not parameters:
                if task.lr is not None:
                    raise ValueError(
                        f"Task '{task.name}' declares lr={task.lr}, but the model has no parameters of "
                        f"its own for it — nothing for the rate to move. Drop the task's lr."
                    )
                continue
            owned.update(id(parameter) for parameter in parameters)
            group: dict[str, Any] = {"params": parameters, "name": task.name}
            if task.lr is not None:
                group["lr"] = task.lr
            groups.append(group)
        shared = [parameter for parameter in self.parameters() if id(parameter) not in owned]
        if shared:
            groups.insert(0, {"params": shared, "name": SHARED_GROUP})
        return groups

    def metric_directions(self) -> dict[str, bool | None]:
        """Each metric's ``higher_is_better`` flag, keyed exactly as this module logs it.

        The structural half of ``DeclaresMetricDirections``: consumers rank
        values without re-deriving semantics from metric names.

        Metrics only. The losses this module also logs are not here because their
        direction needs no declaring — a consumer reads anything undeclared as a
        loss, which is what keeps a runtime part like distillation's from having
        to be announced.
        """
        return {
            log_keys.join(stage, task.name, name): flag
            for task in self._tasks
            for stage, metric_set in task.metrics.items()
            for name, flag in metric_set.directions().items()
        }

    def _fit_profile(self) -> FitProfile:
        """Read the fit-time facts a scheduler may need; only the trainer has them.

        Everything is derived from ``estimated_stepping_batches``: it is the one
        value Lightning guarantees at this point (``num_training_batches`` is
        still infinite until the loops are set up).
        """
        return FitProfile(
            total_steps=int(self.trainer.estimated_stepping_batches),
            epochs=max(int(self.trainer.max_epochs or 0), 1),
        )

    def _shared_step(self, batch: Batch, stage: Stage) -> LightningStepOutput:
        result = self.model.step(batch)
        self._log_losses(result.loss, stage, self._batch_size(batch))
        for task in self._tasks:
            metric_set = task.metrics.get(stage)
            predicted = result.prediction.outputs.get(task.name)
            # Absent rather than empty is a real answer: a family whose head only assembles
            # a decodable output in eval mode produced nothing to evaluate on a training step,
            # and a metric fed a fabricated blank would report that as a score.
            if metric_set is not None and predicted is not None:
                metric_set.update(predicted, result.targets[task.name])
        # Returned, not remembered: Lightning hands a step's return value to every
        # ``on_*_batch_end`` hook, so a consumer is given the batch it was called
        # for and the module keeps no state that could go stale or outlive its use.
        step: LightningStepOutput = {"loss": result.loss.total}
        if self._preview_is_wanted():
            step["preview"] = _preview(result)
        return step

    def _log_losses(self, loss: Loss, stage: Stage, batch_size: int) -> None:
        """The total under ``{stage}/loss``, then every named part beside it.

        Epoch-level throughout: a per-step loss is noise at the resolution anyone
        actually reads a chart at, and ``sync_dist`` makes the epoch value the
        run's rather than rank zero's. Only the total reaches the progress bar,
        and only while training — the parts are what a report is for.
        """
        self.log(
            log_keys.total_loss(stage),
            loss.total,
            prog_bar=stage is Stage.TRAIN,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        for part, value in loss.parts.items():
            self.log(
                log_keys.join(stage, part),
                value,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _preview_is_wanted(self) -> bool:
        """Did any consumer ask for this batch's preview before the step ran?

        A preview shares storage with the activated outputs, which Lightning keeps alive
        through the optimizer step — measured: 352 MB for a ``[16, 21, 512, 512]`` batch — so
        it is built only when asked, per step. Outside a ``Trainer`` a bare call is handed everything.
        """
        # ``self.trainer`` raises when the module is not attached, so the private
        # attribute is what asks *whether* it is; the property reads it after.
        if self._trainer is None:
            return True
        # Lightning assigns ``callbacks`` in ``__init__`` rather than declaring it on the class.
        registered = self.trainer.callbacks  # type: ignore[attr-defined]
        return any(isinstance(consumer, AwaitsPreview) and consumer.awaiting_preview for consumer in registered)

    def _shared_epoch_end(self, stage: Stage) -> None:
        for task in self._tasks:
            metric_set = task.metrics.get(stage)
            if metric_set is None:
                continue
            for name, value in metric_set.compute().items():
                report_metric(
                    log_keys.join(stage, task.name, name),
                    value,
                    scalar_log=self.log,
                    loggers=self.loggers,
                    step=self.current_epoch,
                    class_names=task.class_names,
                )
            metric_set.reset()

    @staticmethod
    def _batch_size(batch: Batch) -> int:
        return len(next(iter(batch.inputs.values())))


def _preview(result: StepResult) -> StepPreview:
    """Detach what a display can use; leave the graph, the loss and the features behind."""
    return StepPreview(
        outputs={name: value.detach() for name, value in result.prediction.outputs.items()},
        targets={name: value.detach() for name, value in result.targets.items()},
    )
