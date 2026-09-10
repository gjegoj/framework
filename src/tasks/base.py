"""Task semantics; class declarations serve assembly, instances receive ready dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import isfinite
from typing import ClassVar, cast

from torch import nn

from src.core import Batch, LossOutput, ModelOutput, ShapeTree, TargetInfo, TensorTree
from src.core.entities import validate_name
from src.losses import Loss, LossInput


class Task(nn.Module, ABC):
    default_head: ClassVar[str | Mapping[str, object] | None] = None
    default_loss: ClassVar[str | Mapping[str, object] | list[Mapping[str, object]] | None] = None
    default_target_encoder: ClassVar[str | Mapping[str, object] | None] = None
    default_metrics: ClassVar[Mapping[str, object]] = {}

    def __init__(
        self, name: str, target_info: TargetInfo, *, output: str | None, loss: Loss | None = None, weight: float = 1.0
    ) -> None:
        super().__init__()
        validate_name(name)
        if not isfinite(weight) or weight <= 0:
            raise ValueError("Task weight must be positive and finite.")
        self.name = name
        self.target_info = target_info
        self.output = output
        self.loss = loss
        self.weight = weight

    @classmethod
    @abstractmethod
    def output_shape(cls, target_info: TargetInfo) -> ShapeTree:
        """Describe the required output dimensions; head adapters translate shapes to library arguments."""
        raise NotImplementedError

    def compute_loss(self, model_output: ModelOutput, batch: Batch) -> LossOutput:
        """Unweighted objective; strategy applies task weight once."""
        if self.loss is None:
            raise ValueError(f"Task {self.name!r} needs a loss to compute its objective.")
        inputs = LossInput(
            outputs=self.select_output(model_output),
            targets=batch.targets.get(self.name),
            model_losses=model_output.losses,
        )
        return cast(LossOutput, self.loss(inputs))

    def select_output(self, model_output: ModelOutput) -> TensorTree:
        """Read one named output, or all outputs for a joint objective such as contrastive loss."""
        return model_output.outputs if self.output is None else model_output.outputs[self.output]

    def postprocess(self, model_output: ModelOutput, batch: Batch) -> object:
        """Stateless interpretation; independent of stage, targets, reporting and self.training."""
        return self.select_output(model_output)

    def prepare_metric_targets(self, batch: Batch) -> object:
        """Metric view of targets; None means no target is available."""
        return batch.targets.get(self.name)
