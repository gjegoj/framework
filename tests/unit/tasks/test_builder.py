"""``build_task_components``: from a universal ``Task`` plus profile to composite bricks."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.core import (
    Backbone,
    Batch,
    DataProfile,
    Features,
    Objective,
    Topology,
)
from src.models import CompositeModel, ExpandedHead, TaskComponents
from src.tasks import build_task_components
from tests.support.entities import a_task, profiling
from tests.support.fakes import FakeEncoder, FlattenBackbone
from tests.support.narrowing import tensor


def test_builds_components_for_a_classification_task() -> None:
    components = build_task_components(a_task(), profiling(label=3), FlattenBackbone(dim=12))

    assert isinstance(components, TaskComponents)
    assert components.head(torch.zeros(2, 12)).shape == (2, 3)
    assert components.weight == 1.0


def test_task_weight_flows_into_the_components() -> None:
    components = build_task_components(a_task(weight=0.5), profiling(label=3), FlattenBackbone(dim=12))

    assert components.weight == 0.5


def test_missing_num_classes_names_the_task_and_hints_setup() -> None:
    with pytest.raises(LookupError, match="label"):
        build_task_components(a_task(), DataProfile(), FlattenBackbone(dim=12))


def test_incompatible_axes_are_rejected_with_both_names() -> None:
    dense_metric = a_task(topology=Topology.DENSE, objective=Objective.METRIC)
    with pytest.raises(ValueError, match="cannot be supervised"):
        build_task_components(dense_metric, profiling(label=3), FlattenBackbone(dim=12))


class TwoStreamBackbone(Backbone):
    """Exposes a second stream and a native head for it."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        flat = inputs["image"].flatten(start_dim=1)
        return Features(streams={"features": flat, "extra": flat * 2})

    def feature_dim(self, stream: str) -> int:
        return 12

    def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
        if stream == "extra":
            return nn.Linear(in_features, out_features)
        return None


def test_stream_override_reads_another_stream() -> None:
    components = build_task_components(a_task(), profiling(label=3), TwoStreamBackbone(), stream="extra")

    assert components.stream == "extra"


def test_prefer_native_head_uses_the_backbones_head() -> None:
    components = build_task_components(
        a_task(), profiling(label=3), TwoStreamBackbone(), stream="extra", prefer_native_head=True
    )

    assert components.head(torch.zeros(2, 12)).shape == (2, 3)


def test_prefer_native_head_fails_loud_when_unavailable() -> None:
    with pytest.raises(LookupError, match="native"):
        build_task_components(a_task(), profiling(label=3), FlattenBackbone(dim=12), prefer_native_head=True)


class DecoderBackbone(Backbone):
    """A dense-capable fake: exposes a decoder stream with spatial dims."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        image = inputs["image"]
        decoder = image.repeat(1, 4, 1, 1)  # [B, 12, H, W] from [B, 3, H, W]
        return Features(streams={"decoder": decoder})

    def feature_dim(self, stream: str) -> int:
        return 12


def test_a_contrastive_task_builds_and_steps_without_targets() -> None:
    """Metric learning closes: two encoders, stacked embeddings, InfoNCE, no target column."""
    from src.models import MultiEncoderBackbone

    torch.manual_seed(0)
    backbone = MultiEncoderBackbone(
        encoders={"image": FakeEncoder("image", 4), "text": FakeEncoder("text", 6)},
        embedding_dim=8,
    )
    task = a_task(name="pair", topology=Topology.MULTISTREAM, objective=Objective.METRIC)

    components = build_task_components(task, DataProfile(), backbone)
    model = CompositeModel(backbone=backbone, components={"pair": components})
    batch = Batch(
        inputs={"image": torch.randn(4, 5), "text": torch.randn(4, 7)},
        targets={},
    )

    loss, prediction, targets = model.step(batch)

    assert set(loss.parts) == {"pair/infonce"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["pair"]).shape == (4, 2, 8)
    assert tensor(targets["pair"]).numel() == 0


def test_a_segmentation_task_builds_and_steps_end_to_end() -> None:
    """DENSE closes: preset -> conv head sized from the decoder -> ce loss on masks."""
    torch.manual_seed(0)
    profile = profiling(label=3)
    task = a_task(topology=Topology.DENSE)
    backbone = DecoderBackbone()

    components = build_task_components(task, profile, backbone)
    model = CompositeModel(backbone=backbone, components={"label": components})
    batch = Batch(
        inputs={"image": torch.randn(2, 3, 8, 8)},
        targets={"label": torch.randint(0, 3, (2, 8, 8))},
    )

    loss, prediction, _ = model.step(batch)

    assert set(loss.parts) == {"label/ce"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["label"]).shape == (2, 3, 8, 8)


def test_built_components_run_inside_a_composite_model() -> None:
    """The full chain closes: data facts, task declarations, model, one step."""
    torch.manual_seed(0)
    backbone = FlattenBackbone(dim=12)
    profile = profiling(label=3)
    tasks = [
        a_task(),
        a_task(name="score", objective=Objective.CONTINUOUS, weight=0.5),
    ]

    components = {task.name: build_task_components(task, profile, backbone) for task in tasks}
    model = CompositeModel(backbone=backbone, components=components)
    batch = Batch(
        inputs={"image": torch.randn(4, 3, 2, 2)},
        targets={"label": torch.tensor([0, 1, 2, 0]), "score": torch.rand(4)},
    )

    loss, prediction, _ = model.step(batch)

    assert set(loss.parts) == {"label/ce", "score/mse"}
    assert loss.total.requires_grad
    assert tensor(prediction.outputs["label"]).shape == (4, 3)
    assert tensor(prediction.outputs["score"]).shape == (4,)


def test_a_native_head_that_already_is_a_head_stays_unwrapped() -> None:
    """Freeze configs name ``model.heads.<task>.base``; a wrapper would bury the
    contract path under a private attribute."""

    class HeadOfferingBackbone(TwoStreamBackbone):
        def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
            return ExpandedHead(base=nn.Linear(in_features, 2), novel=nn.Linear(in_features, 1))

    components = build_task_components(a_task(), profiling(label=3), HeadOfferingBackbone(), prefer_native_head=True)

    assert isinstance(components.head, ExpandedHead)
