"""Shared test fakes: minimal port implementations several test modules exercise.

``test_ports.py`` deliberately keeps its own local copies — its fakes double
as the executable documentation of the ports and must stay self-contained.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics import Metric

from src.core import Backbone, Batch, Features, MetricSet, Model, StepResult, Stream, TaskOutput, as_tensor


class PredictOnlyModel(Model):
    """A model that predicts and refuses to step — the shape every export test needs.

    Export never steps a model: only ``predict`` is on the deployment path. So a test
    about a written graph declares the one method that decides what the graph computes,
    and inherits the refusal that says why the other is missing.
    """

    def step(self, batch: Batch) -> StepResult:
        raise NotImplementedError("Export never steps a model; only predict is on the deployment path.")


class PageLogger(L.pytorch.loggers.Logger):
    """A logger whose whole job is to receive pages — the ``HtmlLogger`` port, structurally."""

    def __init__(self) -> None:
        super().__init__()
        self.pages: list[tuple[str, str, int]] = []

    @property
    def name(self) -> str:
        return "page"

    @property
    def version(self) -> str:
        return "0"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None: ...

    def log_html(self, title: str, html: str, iteration: int) -> None:
        self.pages.append((title, html, iteration))


class FlattenBackbone(Backbone):
    """Flattens the ``image`` input into one features stream of a fixed width."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={Stream.FEATURES: inputs["image"].flatten(start_dim=1)})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: self._dim}


class LearningBackbone(FlattenBackbone):
    """A backbone with weights of its own, for tests about how parameters are grouped.

    ``FlattenBackbone`` has none, and a group holding no parameters is never made —
    which is right, and leaves nothing to check in a test whose whole subject is
    the group everything no task claims belongs to.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.project = torch.nn.Linear(dim, dim)

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        flattened = super().forward(inputs).streams[Stream.FEATURES]
        return Features(streams={Stream.FEATURES: self.project(flattened)})


class FakeEncoder(Backbone):
    """Reads one named input and emits a constant features stream of a fixed width."""

    def __init__(self, input_name: str, dim: int) -> None:
        super().__init__()
        self._input_name = input_name
        self._dim = dim

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        batch = inputs[self._input_name].shape[0]
        return Features(streams={Stream.FEATURES: inputs[self._input_name].new_ones(batch, self._dim)})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: self._dim}


class CountingMetricSet(MetricSet):
    """Counts seen predictions — a deterministic ``MetricSet`` stand-in."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        self.seen += as_tensor(predictions, task="counted", wanted_by="this fake").shape[0]

    def compute(self) -> dict[str, Any]:
        return {"seen": float(self.seen)}

    def reset(self) -> None:
        self.seen = 0

    def directions(self) -> dict[str, bool | None]:
        return {"seen": None}


class SizedMetric(Metric):
    """A metric naming ``num_classes`` — proves derived facts reach imported metrics."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        """Nothing to accumulate; construction is what tests care about."""

    def compute(self) -> Tensor:
        return torch.tensor(0.0)


class Batches(Dataset[Batch]):
    """Already-collated batches handed to a loader with ``batch_size=None``.

    A test that needs Lightning to drive real hooks needs a `Dataset`, and what
    this framework's loop consumes is a ``Batch`` — so the batches are the items
    and collation is skipped rather than re-implemented for a fixture.
    """

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> Batch:
        return self._batches[index]
