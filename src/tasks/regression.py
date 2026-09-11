"""A number per sample, learned either directly or as a distribution over the bins it was cut into."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor

from src.core import CLASS_AXIS, Batch, ModelOutput, TargetInfo, TensorTree, drop_class_axis
from src.tasks.base import LossDeclaration, Task
from src.tasks.registry import task_registry


@task_registry.register("regression")
class Regression(Task):
    """One number per sample. Which of its two forms is in play follows from the target encoder.

    A plain encoder yields the number itself, learned against a distance. A binned one reports the value
    each bin stands for, and the same number is then learned as a distribution over those bins and read
    back as the value it averages to. Either way the metrics compare numbers, so a run swaps between the
    two by changing its encoder and nothing else.
    """

    default_target_encoder: ClassVar[str | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {"mae": {"name": "mae"}}

    @property
    def binned(self) -> tuple[float, ...] | None:
        """The value each bin stands for, when the encoder laid the target out in bins."""
        return self.info.values

    @classmethod
    def out_features(cls, info: TargetInfo) -> int:
        return 1 if info.values is None else len(info.values)

    @property
    def default_loss(self) -> LossDeclaration:
        if self.binned is None:
            return "mse"
        # Cross-entropy leads: it shapes the whole distribution. The expectation term is a correction that
        # keeps the reported number aligned with the metric, and is deliberately the lighter of the two —
        # on its own, any distribution with the right mean would satisfy it.
        return [{"loss": "cross_entropy", "weight": 1.0}, {"loss": "expectation", "weight": 0.5}]

    def loss_target(self, batch: Batch) -> Tensor:
        return self.target(batch).float()

    def metric_view(self, batch: Batch) -> Tensor:
        """A metric compares numbers: for a binned target that is the value its distribution averages to."""
        target = self.target(batch).float()
        return target if self.binned is None else target @ _values(self.binned, target)

    def postprocess(self, output: ModelOutput) -> TensorTree:
        scores = self.raw(output)
        if self.binned is None:
            return drop_class_axis(scores)
        return scores.softmax(dim=CLASS_AXIS) @ _values(self.binned, scores)


def _values(bins: tuple[float, ...], like: Tensor) -> Tensor:
    """The number each bin stands for, on the tensor's own device."""
    return torch.as_tensor(bins, dtype=torch.float, device=like.device)
