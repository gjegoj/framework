"""Values exchanged by data, models, tasks and reporting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import cast

import torch
from torch import Tensor

from src.core.types import ShapeTree, TensorTree, tree_map


@dataclass(frozen=True, slots=True)
class Sample:
    inputs: Mapping[str, object]
    targets: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    auxiliary_inputs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Batch:
    """The collator supplies example count; ragged tensor lengths do not determine it."""

    inputs: Mapping[str, TensorTree]
    count: int = field(kw_only=True)
    targets: Mapping[str, TensorTree] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Batch count must be a positive integer.")

    def __len__(self) -> int:
        return self.count

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> Batch:
        return self._map(lambda tensor: tensor.to(device=device, non_blocking=non_blocking))

    def detach(self) -> Batch:
        return self._map(Tensor.detach)

    def _map(self, operation: Callable[[Tensor], Tensor]) -> Batch:
        return Batch(
            inputs=cast(Mapping[str, TensorTree], tree_map(operation, self.inputs)),
            targets=cast(Mapping[str, TensorTree], tree_map(operation, self.targets)),
            count=len(self),
            metadata=self.metadata,
        )


def validate_classes(classes: Mapping[int, str]) -> None:
    """Validate the user-defined output vocabulary, never infer it from observations."""
    if not classes or any(type(index) is not int for index in classes):
        raise ValueError("Classes require integer indices.")
    if set(classes) != set(range(len(classes))):
        raise ValueError("Class indices must be contiguous from zero.")
    if any(not isinstance(name, str) or not name.strip() for name in classes.values()):
        raise ValueError("Class names must be nonblank strings.")


@dataclass(frozen=True, slots=True)
class Normalization:
    """Per-channel mean and std an input is trained with: the pixel pipeline applies it, export ships it."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean or len(self.mean) != len(self.std):
            raise ValueError("Normalization needs one mean and one std per channel.")
        if any(not isfinite(value) for value in (*self.mean, *self.std)) or any(value <= 0 for value in self.std):
            raise ValueError("Normalization needs finite means and positive stds.")


@dataclass(frozen=True, slots=True)
class InputInfo:
    shape: ShapeTree
    modality: str | None = None
    normalization: Normalization | None = None


@dataclass(frozen=True, slots=True)
class TargetInfo:
    """Resolved facts about one target: its per-sample shape, its vocabulary, and the number each class stands for."""

    shape: ShapeTree = None
    classes: Mapping[int, str] | None = None
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.classes is not None:
            validate_classes(self.classes)
        if self.values is not None:
            if self.classes is None or len(self.values) != len(self.classes):
                raise ValueError("Target values stand one behind each class index; declare both, same length.")
            if any(not isfinite(value) for value in self.values):
                raise ValueError("Target values must be finite numbers.")

    @property
    def num_classes(self) -> int | None:
        return None if self.classes is None else len(self.classes)


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    inputs: Mapping[str, InputInfo]
    targets: Mapping[str, TargetInfo]
    splits: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)


type Prediction = Mapping[str, TensorTree]
"""What a task's output means, per task: the same tree every other value in a batch is."""


CONTRIBUTION = "contribution"
"""The leaf a term's weighted share is reported under, beside the term reporting itself.

A leaf rather than a prefix: a composed name reads ``<task>/<term>``, and a namespace in front of it
would take the task's place — the share of `mask/dice` is `mask/dice/contribution`.
"""


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Raw losses never change under weighting; contributions describe the actual objective."""

    total: Tensor
    losses: Mapping[str, Tensor] = field(default_factory=dict)
    contributions: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total.ndim != 0:
            raise ValueError("Loss total must be a scalar tensor.")

    def __add__(self, other: LossOutput) -> LossOutput:
        duplicates = (self.losses.keys() & other.losses.keys()) | (
            self.contributions.keys() & other.contributions.keys()
        )
        if duplicates:
            raise ValueError(
                f"Two loss terms report under the same name: {', '.join(sorted(duplicates))}. Give one a "
                "log_name of its own, or check that each task prefixes what it reports."
            )
        return LossOutput(
            self.total + other.total, {**self.losses, **other.losses}, {**self.contributions, **other.contributions}
        )

    def __mul__(self, weight: float) -> LossOutput:
        if not isfinite(weight):
            raise ValueError("Loss weight must be finite.")
        if weight == 1.0:
            # A weight of one is not a weighting: the same values under both names, so a report has
            # nothing to show twice and a step does no arithmetic to arrive back where it started.
            return self
        return LossOutput(
            self.total * weight, self.losses, {name: value * weight for name, value in self.contributions.items()}
        )

    __rmul__ = __mul__

    def prefixed(self, prefix: str) -> LossOutput:
        """Namespace loss values before combining tasks; tracking adds stage and split."""
        return LossOutput(
            self.total,
            {f"{prefix}{SEGMENT}{name}": value for name, value in self.losses.items()},
            {f"{prefix}{SEGMENT}{name}": value for name, value in self.contributions.items()},
        )

    def breakdown(self) -> Mapping[str, Tensor]:
        """What a report shows besides the total: every term as itself, plus its share where a weight moved it.

        The two readings answer different questions — a term compares across runs whatever weights they
        gave it, a share explains which of them the total is made of — and a run that weighs nothing sees
        only the first. Told apart by identity rather than by value: weighting returns new tensors and
        not weighting returns the very ones reported, so the answer costs no arithmetic per step.
        """
        shares = {
            f"{name}{SEGMENT}{CONTRIBUTION}": value
            for name, value in self.contributions.items()
            if value is not self.losses.get(name)
        }
        return {**self.losses, **shares}


@dataclass(frozen=True, slots=True)
class ModelOutput:
    outputs: Mapping[str, TensorTree] = field(default_factory=dict)
    features: Mapping[str, TensorTree] = field(default_factory=dict)


@dataclass(slots=True)
class Matrix:
    """A two-dimensional reading with its axes named by the metric that knew them.

    Which axis holds the prediction cannot be read off the tensor, and a chart drawn the other way
    round is a plausible-looking lie, so the metric states it here rather than leaving it to be guessed.

    ``labels`` name the rows and columns where something knows what they stand for. A metric counts and
    has no vocabulary, so they are filled in on the way to a tracker, by whoever holds the task.

    The one value here that is not frozen, and not by choice: measured on torchmetrics 1.9.0, a
    collection runs whatever ``compute`` returns through ``apply_to_collection``, which rebuilds a
    dataclass field by field and raises on a frozen one.
    """

    value: Tensor
    xaxis: str
    yaxis: str
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.value.ndim != 2:
            raise ValueError(f"A matrix is drawn from two axes; this reading has {self.value.ndim}.")
        if self.labels is not None and len(self.labels) != self.value.shape[0]:
            raise ValueError(
                f"A matrix is labelled row by row: {self.value.shape[0]} rows against {len(self.labels)} labels."
            )


@dataclass(frozen=True, slots=True)
class StepOutput:
    """Training requires loss; evaluation may report metrics without a loss."""

    loss: LossOutput | None
    predictions: Prediction = field(default_factory=dict)
    targets: Mapping[str, TensorTree] = field(default_factory=dict)


SEGMENT = "/"
"""What joins the parts of a composed name: a task and its term, a family and its leaf."""

NAME_SEPARATORS = f".{SEGMENT}"
"""Characters a name may not contain: a dot addresses modules, and the other composes the names above."""


def validate_name(name: str, *, label: str = "Component") -> None:
    if not name or name.strip() != name or any(char in name for char in NAME_SEPARATORS):
        raise ValueError(f"{label} names must be nonblank, unpadded and free of {NAME_SEPARATORS!r}: {name!r}.")
