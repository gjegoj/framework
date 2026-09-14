"""One specimen per registered task kind: the task, a batch it could be given, and an output for it.

Every kind a config may name is exercised against the same three values, so a new kind is one row here
and nothing else: the tests that read this table hold tasks and metrics to the same shapes.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.core import Axis, Batch, ModelOutput, Representation, TargetInfo, require_tensor
from src.tasks import Task
from src.tasks.registry import task_registry

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
COUNT = 3
EXTENT = {Axis.HEIGHT: 4, Axis.WIDTH: 5}
"""How many samples a specimen batch holds, and the extent a dense prediction has where its shape is open."""

WIDTH = 4
"""How wide a specimen embedding is — a choice of the model, so a kind that has one declares it."""

PIXELS = torch.arange(COUNT * EXTENT[Axis.HEIGHT] * EXTENT[Axis.WIDTH]).reshape(
    COUNT, EXTENT[Axis.HEIGHT], EXTENT[Axis.WIDTH]
)
"""A running number per pixel, so a dense target below can be cut into as many classes as its kind has."""


def info(**overrides: Any) -> TargetInfo:
    """The three-class vocabulary the specimens share, with whatever a test changes about it."""
    return TargetInfo(**{"classes": CLASSES, **overrides})


ON_THE_KIND: dict[str, dict[str, Any]] = {
    "metric_learning": {"embedding_dim": WIDTH},
    "contrastive": {"embedding_dim": WIDTH},
}
"""What a kind takes in its own declaration, where the data cannot settle it: read by ``specimen`` alone."""

SPECIMENS: dict[str, tuple[TargetInfo, Tensor]] = {
    "classification": (info(), torch.tensor([0, 1, 2])),
    "binary_classification": (TargetInfo(), torch.tensor([0.0, 1.0, 0.0])),
    "multilabel_classification": (info(), torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])),
    "segmentation": (info(), PIXELS % len(CLASSES)),
    "binary_segmentation": (TargetInfo(), (PIXELS % 2).float()),
    "regression": (TargetInfo(), torch.tensor([1.5, 2.5, 3.5])),
    "metric_learning": (info(), torch.tensor([0, 1, 0])),
    # The one kind that reads no column: what stands here is what it derives from the batch, which is
    # which row each sample is, so the table still says "the target this kind is handed".
    "contrastive": (TargetInfo(), torch.arange(COUNT)),
}
"""What each kind's target says and how its own encoder hands it over: a new kind needs a row here.

Every class of a kind's vocabulary appears in its target, so a metric normalised over the true classes
has no empty row to divide by — a specimen leaving a class unseen would measure a case no split has.

The one row that breaks that on purpose is the kind judged by retrieval: its readings rank samples
against each other rather than normalise over a vocabulary, and what they need instead is an identity
that *repeats*, since a picture with nobody of its own to find is left out of them.
"""


def published(task: Task, output: ModelOutput) -> Tensor:
    """What a task answers with for a plain projection, which is what every specimen output is.

    One home for the three suites that ask it — tasks, metrics and the page — so a change to what a
    task is handed is one line here rather than a search for every caller of ``postprocess``.
    """
    return require_tensor(task.postprocess(output, Representation.PROJECTED), name=task.name)


ANSWERS_PER_SAMPLE: dict[str, int] = {"contrastive": 2}
"""How many rows a kind answers with for each sample of the batch, where that is not one.

Every other kind answers once per sample, so a batch of three has three rows. A run that draws views of
a picture has as many answers as draws: the stage stacks them and a viewing backbone folds them into the
batch, which is what makes one set of weights see every draw. ``Batch.count`` keeps counting samples, so
the two numbers part company here rather than anywhere in a declaration.
"""


def specimen(kind: str, name: str = "t") -> tuple[Task, ModelOutput, Batch]:
    """A task of one kind, an output its head could have produced, and the batch that output answers."""
    if kind not in SPECIMENS:
        raise LookupError(f"{kind!r} is registered but has no specimen; add a row to SPECIMENS in {__name__}.")
    declared, target = SPECIMENS[kind]
    task = task_registry.get(kind)(name, declared, **ON_THE_KIND.get(kind, {}))
    shape = task.output_shape()
    declared_sizes = zip(shape.axes, shape.sizes, strict=True)
    sizes = [size if size is not None else EXTENT[Axis(axis)] for axis, size in declared_sizes]
    output = ModelOutput(outputs={name: torch.rand(COUNT * ANSWERS_PER_SAMPLE.get(kind, 1), *sizes)})
    return task, output, Batch(inputs={}, targets={name: target}, count=COUNT)
