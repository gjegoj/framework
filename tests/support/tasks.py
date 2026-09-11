"""One specimen per registered task kind: the task, a batch it could be given, and an output for it.

Every kind a config may name is exercised against the same three values, so a new kind is one row here
and nothing else: the tests that read this table hold tasks and metrics to the same shapes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.core import Axis, Batch, ModelOutput, TargetInfo
from src.tasks import Task
from src.tasks.registry import task_registry

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
COUNT = 3
EXTENT = {Axis.HEIGHT: 4, Axis.WIDTH: 5}
"""How many samples a specimen batch holds, and the extent a dense prediction has where its shape is open."""

PIXELS = torch.arange(COUNT * EXTENT[Axis.HEIGHT] * EXTENT[Axis.WIDTH]).reshape(
    COUNT, EXTENT[Axis.HEIGHT], EXTENT[Axis.WIDTH]
)
"""A running number per pixel, so a dense target below can be cut into as many classes as its kind has."""


def info(**overrides: Any) -> TargetInfo:
    """The three-class vocabulary the specimens share, with whatever a test changes about it."""
    return TargetInfo(**{"classes": CLASSES, **overrides})


SPECIMENS: dict[str, tuple[TargetInfo, Tensor]] = {
    "classification": (info(), torch.tensor([0, 1, 2])),
    "binary_classification": (TargetInfo(), torch.tensor([0.0, 1.0, 0.0])),
    "multilabel_classification": (info(), torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])),
    "segmentation": (info(), PIXELS % len(CLASSES)),
    "binary_segmentation": (TargetInfo(), (PIXELS % 2).float()),
    "regression": (TargetInfo(), torch.tensor([1.5, 2.5, 3.5])),
}
"""What each kind's target says and how its own encoder hands it over: a new kind needs a row here.

Every class of a kind's vocabulary appears in its target, so a metric normalised over the true classes
has no empty row to divide by — a specimen leaving a class unseen would measure a case no split has.
"""


def specimen(kind: str, name: str = "t") -> tuple[Task, ModelOutput, Batch]:
    """A task of one kind, an output its head could have produced, and the batch that output answers."""
    if kind not in SPECIMENS:
        raise LookupError(f"{kind!r} is registered but has no specimen; add a row to SPECIMENS in {__name__}.")
    declared, target = SPECIMENS[kind]
    task = task_registry.get(kind)(name, declared)
    shape = task.output_shape(declared)
    declared_sizes = zip(shape.axes, shape.sizes, strict=True)
    sizes = [size if size is not None else EXTENT[Axis(axis)] for axis, size in declared_sizes]
    output = ModelOutput(outputs={name: torch.rand(COUNT, *sizes)})
    return task, output, Batch(inputs={}, targets={name: target}, count=COUNT)
