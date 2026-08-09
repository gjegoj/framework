"""Port contracts: abstractness, typed ``__call__``, and an end-to-end step.

The fakes in this module double as executable documentation: minimal
implementations of every port, composed into one training step without
a training loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from src.core import (
    AdaptedTarget,
    Backbone,
    Batch,
    Criterion,
    DataModule,
    Features,
    Head,
    Loss,
    MetricSet,
    Model,
    TargetAdapter,
    TaskOutput,
    as_tensor,
)
from tests.support.narrowing import tensor


class FlattenBackbone(Backbone):
    """Flattens the image into a single feature stream of fixed dimensionality."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={"features": inputs["image"].flatten(start_dim=1)})

    def feature_dims(self) -> Mapping[str, int]:
        return {"features": self._dim}


class LinearHead(Head):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self._linear = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor) -> Tensor:
        # nn.Module.__call__ erases the return type to Any; pin it back.
        return cast(Tensor, self._linear(features))


class CrossEntropyCriterion(Criterion):
    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.part("ce", nn.functional.cross_entropy(logits, target))


class CountingMetricSet(MetricSet):
    def __init__(self) -> None:
        super().__init__()
        self.seen = 0

    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        self.seen += as_tensor(predictions, task="counted", wanted_by="this fake").shape[0]

    def compute(self) -> dict[str, Any]:
        return {"seen": self.seen}

    def reset(self) -> None:
        self.seen = 0

    def directions(self) -> dict[str, bool | None]:
        return {"seen": None}


@pytest.mark.parametrize("port", [Backbone, Head, Criterion, MetricSet, DataModule, Model])
def test_every_port_is_abstract(port: type[Any]) -> None:
    with pytest.raises(TypeError):
        port()


def test_backbone_call_is_typed_and_still_runs_module_hooks() -> None:
    backbone = FlattenBackbone(dim=12)
    seen: list[type[object]] = []
    backbone.register_forward_hook(lambda module, args, output: seen.append(type(output)))

    features = backbone({"image": torch.zeros(2, 3, 2, 2)})

    assert isinstance(features, Features)
    assert seen == [Features]


def test_an_unknown_stream_is_refused_by_the_port_naming_what_is_offered() -> None:
    """One refusal for every adapter, written where the port already knows the answer.

    Five adapters each branched on the stream and raised their own spelling of this
    sentence — and none of them could be asked what they *do* offer, which is the other
    half of what naming a stream is for. ``feature_dims`` is now the declaration and
    ``feature_dim`` the lookup over it.
    """
    with pytest.raises(LookupError, match=r"exposes 'features', requested 'decoder'"):
        FlattenBackbone(dim=4).feature_dim("decoder")


def test_metric_set_accumulates_and_resets() -> None:
    metrics = CountingMetricSet()
    metrics.update(torch.zeros(4), torch.zeros(4))

    assert metrics.compute() == {"seen": 4}

    metrics.reset()
    assert metrics.compute() == {"seen": 0}


def test_one_training_step_composes_from_ports_alone() -> None:
    """End-to-end step: Batch → Backbone → Head → Criterion/Activation → Loss + metrics."""
    torch.manual_seed(0)
    batch = Batch(
        inputs={"image": torch.randn(4, 3, 2, 2)},
        targets={"label": torch.tensor([0, 1, 2, 0])},
    )
    backbone = FlattenBackbone(dim=12)
    head = LinearHead(in_features=12, out_features=3)
    criterion = CrossEntropyCriterion()
    adapt: TargetAdapter = lambda target: AdaptedTarget(for_loss=target, for_metrics=target)
    metrics = CountingMetricSet()

    features = backbone(batch.inputs)
    logits = head(features["features"])
    target = adapt(tensor(batch.targets["label"]))
    loss = criterion(logits, target.for_loss).scoped("label")
    metrics.update(logits.argmax(dim=1), target.for_metrics)

    assert set(loss.parts) == {"label/ce"}
    assert loss.total.requires_grad
    assert metrics.compute() == {"seen": 4}
