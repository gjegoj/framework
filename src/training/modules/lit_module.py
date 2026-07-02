"""LitModule: the standard supervised training module.

Implements ``_shared_step`` on top of :class:`BaseLitModule`: one forward pass, then
per-task loss + metric update, then loss aggregation. All decision logic lives in the Task
objects and the aggregator; this is a thin coordinator. Other regimes (e.g. knowledge
distillation) subclass ``BaseLitModule`` and override ``_shared_step`` instead.

The step loop:

    output = model(batch.inputs)
    for task in tasks:
        logits = output.task_logits[task.name]
        target = task.adapter.adapt(batch.targets[task.name])
        losses[task.name] = task.criterion(logits, target.loss)
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
        ``logits``).
        """
        output = self.model(batch.inputs)
        losses: dict[str, LossResult] = {}
        task_views: dict[str, TaskStepView] = {}

        for task in self.tasks:
            logits = output.task_logits[task.name]
            target = task.adapter.adapt(batch.targets[task.name])
            predictions = task.activation(logits).detach()
            losses[task.name] = task.criterion(logits, target.loss)
            task.metrics[stage].update(predictions, target.metric)
            task_views[task.name] = TaskStepView(predictions=predictions, metric_target=target.metric)

        combined_loss = self._loss_aggregator.combine(losses, self._task_weights)
        self._log_losses(combined_loss, stage)
        return {"loss": combined_loss.total, "task_views": task_views}
