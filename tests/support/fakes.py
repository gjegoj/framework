"""Shared test fakes: minimal port implementations several test modules exercise.

``test_ports.py`` deliberately keeps its own local copies — its fakes double
as the executable documentation of the ports and must stay self-contained.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightning as L
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchmetrics import Metric

from src.core import (
    Backbone,
    Batch,
    Criterion,
    Features,
    Loss,
    Model,
    Prediction,
    StepResult,
    Stream,
    TaskOutput,
    require_tensor,
)
from src.export import DeployableModel, Exporter, Runnable
from src.losses import CrossEntropyCriterion
from src.metrics.ports import MetricSet
from src.models import CompositeModel, LinearHead, TaskComponents
from src.models.composite import Activation
from src.tasks.activations import softmax_probabilities
from src.tasks.adapters import as_class_indices


class OwnLossModel(Model):
    """A model that arrives whole — its own head, its own loss — built as it is from ``model: {_target_: ...}``.

    Pools the image, classifies it, and steps with cross-entropy under the part name
    ``ce``, scoped by the task it serves. Nothing of the composite family is involved,
    which is what the tests using it are about.
    """

    def __init__(self, num_classes: int, task: str = "label") -> None:
        super().__init__()
        self.task = task
        self.classifier = nn.Linear(3, num_classes)

    def step(self, batch: Batch) -> StepResult:
        logits = self._logits(batch)
        target = batch.targets[self.task]
        assert isinstance(target, Tensor)  # a classification target; the fake serves no other kind
        loss = Loss.part("ce", F.cross_entropy(logits, target)).scoped(self.task)
        return StepResult(loss=loss, prediction=self._prediction(logits), targets=dict(batch.targets))

    def predict(self, batch: Batch) -> Prediction:
        return self._prediction(self._logits(batch))

    def _logits(self, batch: Batch) -> Tensor:
        logits: Tensor = self.classifier(batch.inputs["image"].mean(dim=(2, 3)))
        return logits

    def _prediction(self, logits: Tensor) -> Prediction:
        return Prediction(outputs={self.task: logits.softmax(dim=1)}, logits={self.task: logits})


class PredictOnlyModel(Model):
    """A model that predicts and refuses to step — the shape every export test needs.

    Export never steps a model: only ``predict`` is on the deployment path. So a test
    about a written graph declares the one method that decides what the graph computes,
    and inherits the refusal that says why the other is missing.
    """

    def step(self, batch: Batch) -> StepResult:
        raise NotImplementedError("Export never steps a model; only predict is on the deployment path.")


class DoublingModel(PredictOnlyModel):
    """Doubles the 'image' input into task 'label' — a graph that traces and can be checked by hand."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"] * 2})


class PoolingModel(PredictOnlyModel):
    """Averages an image into one number per sample — a model with a shape opinion."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"].mean(dim=(1, 2, 3), keepdim=False)})


class VectorModel(PredictOnlyModel):
    """Reads a flat vector, so an image-shaped example cannot reach it."""

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": batch.inputs["image"] @ torch.ones(7)})


class FakeExporter(Exporter):
    """Writes nothing and loads back whatever the test says the artifact does.

    Verification's job is to compare a runnable against a model; a fake runnable is how
    a test states the drift it wants compared.
    """

    def __init__(self, runnable: Runnable) -> None:
        super().__init__()
        self._runnable = runnable

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        return destination

    def load(self, path: Path) -> Runnable:
        return self._runnable


class PageLogger(L.pytorch.loggers.Logger):
    """A logger whose whole job is to receive pages and tags — the ``HtmlLogger`` and ``TagsRuns`` ports, structurally."""

    def __init__(self) -> None:
        super().__init__()
        self.pages: list[tuple[str, str, int]] = []
        self.tags: list[str] = []

    def tag_run(self, architecture: str | None) -> None:
        if architecture:
            self.tags.append(architecture)

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


def a_composite(
    dim: int = 12,
    *,
    classes: int = 2,
    tasks: tuple[str, ...] = ("label",),
    criterion: Criterion | None = None,
    activation: Activation | None = None,
    backbone: Backbone | None = None,
) -> CompositeModel:
    """One classification head per task on a flattening backbone — what ``build_model`` composes.

    ``LinearHead(dim, classes)``, cross-entropy, softmax and class-index targets per task:
    the model eleven files built by hand before this. A test about the composite itself, or
    about one particular criterion, still builds its own; a test about something *around*
    the model names only what it changes. A ``criterion`` given serves every task named.
    """
    return CompositeModel(
        backbone=backbone if backbone is not None else FlattenBackbone(dim),
        components={
            name: TaskComponents(
                head=LinearHead(dim, classes),
                criterion=criterion if criterion is not None else CrossEntropyCriterion(),
                activation=activation if activation is not None else softmax_probabilities,
                target_adapter=as_class_indices,
            )
            for name in tasks
        },
    )


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
        self.seen += require_tensor(predictions, task="counted", wanted_by="this fake").shape[0]

    def compute(self) -> dict[str, Any]:
        return {"seen": float(self.seen)}

    def reset(self) -> None:
        self.seen = 0

    def directions(self) -> dict[str, bool | None]:
        return {"seen": None}


class SizedMetric(Metric):
    """A metric naming ``num_classes`` — proves an imported metric is sized by signature like a registered one."""

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


def clearml_stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A ``clearml`` module in ``sys.modules`` whose task records what a logger reports to it.

    One stub for every test that builds a ``ClearMLLogger``, because the stub has a
    subtlety: it carries a real ``__spec__``. ``accelerate`` probes for clearml with
    ``importlib.util.find_spec``, which raises rather than answering "no" when a module
    in ``sys.modules`` has none — measured, whether it raised depended on whether
    something else had imported accelerate first. Returns the record.
    """
    recorded = SimpleNamespace(
        scalars=[],
        matrices=[],
        curves=[],
        single_values=[],
        media=[],
        histograms=[],
        figures=[],
        flushed=0,
        init_kwargs=None,
        connected=None,
        added_tags=[],
    )

    class _Backend:
        def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
            recorded.scalars.append((title, series, value, iteration))

        def report_single_value(self, name: str, value: float) -> None:
            recorded.single_values.append((name, value))

        def report_confusion_matrix(self, **kwargs: Any) -> None:
            recorded.matrices.append(kwargs)

        def report_scatter2d(self, **kwargs: Any) -> None:
            recorded.curves.append(kwargs)

        def report_media(self, **kwargs: Any) -> None:
            recorded.media.append(kwargs)

        def report_histogram(self, **kwargs: Any) -> None:
            recorded.histograms.append(kwargs)

        def report_plotly(self, **kwargs: Any) -> None:
            recorded.figures.append(kwargs)

    class _Task:
        name = "run"
        id = "abc123"

        @classmethod
        def init(cls, **kwargs: Any) -> _Task:
            recorded.init_kwargs = kwargs
            return cls()

        def add_tags(self, tags: list[str]) -> None:
            recorded.added_tags.extend(tags)

        def get_logger(self) -> _Backend:
            return _Backend()

        def connect(self, mapping: dict[str, Any]) -> None:
            recorded.connected = mapping

        def flush(self) -> None:
            recorded.flushed += 1

    recorded.task = _Task  # the stub's own class, for a test that makes the backend misbehave
    monkeypatch.setitem(
        sys.modules, "clearml", SimpleNamespace(Task=_Task, __spec__=ModuleSpec("clearml", loader=None))
    )
    return recorded
