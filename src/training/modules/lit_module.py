"""LitModule: the standard supervised training module.

Implements ``_shared_step`` on top of :class:`BaseLitModule`: one forward pass, then
per-task loss + metric update, then loss aggregation. All decision logic lives in the Task
objects and the aggregator; this is a thin coordinator. The loop itself varies only at two
seams — :meth:`BaseLitModule._auxiliary_targets` (extra per-batch targets) and
:meth:`BaseLitModule._task_loss` (how one task's loss is formed) — so a regime like
knowledge distillation subclasses ``LitModule`` and overrides just those hooks rather than
copying the loop.

The step loop:

    output = model(batch.inputs)
    auxiliary = self._auxiliary_targets(batch, stage)   # {} for standard training
    for task in tasks:
        logits = output.task_logits[task.name]
        target = task.adapter.adapt(batch.targets[task.name])
        losses[task.name] = self._task_loss(task, logits, target, auxiliary)
        task.metrics[stage].update(task.activation(logits), target.metric)
    total = aggregator.combine(losses, weights)
"""

from __future__ import annotations

from src.core.entities import Batch, LossResult, StepOutput, TaskStepView
from src.core.enums import Stage
from src.training.modules.base import BaseLitModule


class LitModule(BaseLitModule):
    """Standard supervised multi-task module: forward → per-task loss → aggregate."""

    def _shared_step(self, batch: Batch, stage: Stage) -> StepOutput:
        """Run the forward + loss/metric loop and return step artifacts.

        Returns a :class:`StepOutput` dict. ``task_views`` (post-activation predictions + metric
        targets) flow to ``on_*_batch_end(outputs, ...)`` so visualization callbacks reuse
        step work without re-running activation or adapter adaptation. ``predictions`` are detached —
        the activation output only feeds metrics/inference, never backprop (the loss runs on
        ``logits``). Per-task loss formation is delegated to :meth:`_task_loss` so regimes vary
        the loss without re-implementing this loop.
        """
        output = self.model(batch.inputs)
        auxiliary = self._auxiliary_targets(batch, stage)
        losses: dict[str, LossResult] = {}
        task_views: dict[str, TaskStepView] = {}

        for task in self.tasks:
            logits = output.task_logits[task.name]
            target = task.adapter.adapt(batch.targets[task.name])
            predictions = task.activation(logits).detach()
            losses[task.name] = self._task_loss(task, logits, target, auxiliary)
            task.metrics[stage].update(predictions, target.metric)
            task_views[task.name] = TaskStepView(predictions=predictions, metric_target=target.metric)

        combined_loss = self._loss_aggregator.combine(losses, self._task_weights)
        self._log_losses(combined_loss, stage)
        return {"loss": combined_loss.total, "task_views": task_views}
