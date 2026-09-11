"""What a run learns for one target: the class is the semantics, the instance is this run's facts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import ClassVar

from torch import Tensor

from src.core import Axis, Batch, ModelOutput, Stream, TargetInfo, TensorShape, TensorTree, require_tensor
from src.core.entities import validate_name

type LossDeclaration = str | Mapping[str, object] | Sequence[Mapping[str, object]]
"""A loss as a task declares its default: a registry name, one declaration, or several to weigh together."""


class Task(ABC):
    """One learning objective: its semantics as a subclass, its name, facts and weight as an instance.

    A task holds no modules. It declares what a run should assemble around it — which encoder reads its
    column, which head serves it, what it is judged by — and it converts between the three views one step
    needs: what the loss compares, what the model's output means, and what a metric scores.

    Attributes:
        default_head: The head a task gets when a run declares none, and the stream it reads.
        default_target_encoder: Registry name of the encoder its column starts from, or None when the
            batch itself is the supervision. Read before the data is prepared, so it cannot see facts.
        default_metrics: What the task is judged by when a run declares no metrics of its own.
    """

    default_head: ClassVar[Mapping[str, object]] = {"name": "linear", "input": Stream.POOLED}
    default_target_encoder: ClassVar[str | None] = None
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {}

    def __init__(self, name: str, info: TargetInfo, *, weight: float = 1.0) -> None:
        validate_name(name, kind="Task")
        if not isfinite(weight) or weight <= 0:
            raise ValueError(f"Task {name!r} needs a positive, finite weight; got {weight}.")
        self.name = name
        self.info = info
        self.weight = weight

    @classmethod
    @abstractmethod
    def out_features(cls, info: TargetInfo) -> int:
        """How many values the model produces per position — per sample, or per pixel for a dense task."""

    @classmethod
    def output_shape(cls, info: TargetInfo) -> TensorShape:
        """The shape one prediction has, which is what a head is built to produce."""
        return TensorShape(axes=(Axis.CLASSES,), sizes=(cls.out_features(info),))

    @property
    @abstractmethod
    def default_loss(self) -> LossDeclaration:
        """The objective this task is learned by when a run declares none; instance-level, so it can read facts."""

    def metric_kwargs(self) -> Mapping[str, object]:
        """Facts a metric's constructor may name — the vocabulary it scores against."""
        return {}

    @abstractmethod
    def loss_target(self, batch: Batch) -> Tensor:
        """The target as the loss compares it; a batch transform may have left it soft."""

    @abstractmethod
    def metric_view(self, batch: Batch) -> Tensor:
        """The same target as a metric scores it — hard where the loss reads a share."""

    @abstractmethod
    def postprocess(self, output: ModelOutput) -> TensorTree:
        """What the model's raw output means: probabilities, a value, a mask. Never the loss's view."""

    def target(self, batch: Batch) -> Tensor:
        """This task's raw target, refused by name when the batch carries none."""
        return self._own(batch.targets, "target in the batch")

    def raw(self, output: ModelOutput) -> Tensor:
        """What this task's head produced, before it means anything."""
        return self._own(output.outputs, "output from the model")

    def _own(self, values: Mapping[str, TensorTree], what: str) -> Tensor:
        try:
            return require_tensor(values[self.name], name=self.name)
        except KeyError:
            held = ", ".join(sorted(values)) or "none"
            raise LookupError(f"No {what} for task {self.name!r}; there is: {held}.") from None
