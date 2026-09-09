"""Every registered backbone, against the port: the streams it names are the streams it produces.

Each family's own file proves its particulars; this one proves what the composition root
relies on for any of them, parametrized over the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Backbone
from src.models import HFTextBackbone, MultiEncoderBackbone, MultiViewBackbone, SmpBackbone, TimmBackbone
from src.models.registry import backbone_registry
from tests.support.fakes import FakeEncoder

TINY_BERT = "hf-internal-testing/tiny-random-bert"


def ultralytics() -> Backbone:
    pytest.importorskip("ultralytics", reason="the detection family parses ultralytics' graphs")
    from src.models.backbones.ultralytics import UltralyticsBackbone

    return UltralyticsBackbone("yolov8n.yaml")


SHIPPED: dict[str, tuple[Callable[[], Backbone], dict[str, Tensor]]] = {
    "timm": (lambda: TimmBackbone(model_name="resnet18", pretrained=False), {"image": torch.randn(2, 3, 32, 32)}),
    "smp": (lambda: SmpBackbone(pretrained=False), {"image": torch.randn(2, 3, 32, 32)}),
    "hf_text": (
        lambda: HFTextBackbone(model_name=TINY_BERT, pretrained=False),
        {"text": torch.randint(1, 30, (2, 12))},
    ),
    "multi": (
        lambda: MultiEncoderBackbone(
            encoders={"image": FakeEncoder("image", 4), "text": FakeEncoder("text", 6)}, embedding_dim=8
        ),
        {"image": torch.randn(3, 5), "text": torch.randn(3, 7)},
    ),
    "multiview": (lambda: MultiViewBackbone(inner=FakeEncoder("image", 4)), {"image": torch.randn(2, 3, 5)}),
    "ultralytics": (ultralytics, {"image": torch.zeros(2, 3, 64, 64)}),
}
"""Name → the smallest construction that needs no download, and an input it reads."""

HUB = {"hf_text"}
"""Built from hub-shaped weights; outside the gate under the hub marker."""

NAMES = [pytest.param(name, marks=pytest.mark.slow) if name in HUB else name for name in sorted(SHIPPED)]


def test_every_registered_backbone_has_a_row_here() -> None:
    assert set(map(str, backbone_registry)) == set(SHIPPED)


@pytest.mark.parametrize("name", NAMES)
def test_feature_dims_names_exactly_the_streams_forward_produces(name: str) -> None:
    """The widths a head is sized from are the widths on the channel axis of what arrives."""
    build, inputs = SHIPPED[name]
    backbone = build().eval()  # shapes, not training: BatchNorm has no say over a batch of two

    features = backbone(inputs)

    assert set(features.streams) == set(backbone.feature_dims())
    for stream, width in backbone.feature_dims().items():
        assert width in {features[stream].shape[1], features[stream].shape[-1]}, (name, stream)


@pytest.mark.parametrize("name", NAMES)
def test_every_pyramid_level_has_a_width(name: str) -> None:
    build, _ = SHIPPED[name]
    backbone = build()

    for level in backbone.pyramid():
        assert backbone.feature_dim(level) > 0


@pytest.mark.parametrize("name", NAMES)
def test_an_unknown_stream_is_refused_naming_the_streams_it_has(name: str) -> None:
    build, _ = SHIPPED[name]
    backbone = build()

    with pytest.raises(LookupError) as refused:
        backbone.feature_dim("no_such_stream")

    assert all(stream in str(refused.value) for stream in backbone.feature_dims())


@pytest.mark.parametrize("name", NAMES)
def test_a_native_head_is_a_module_or_none(name: str) -> None:
    build, _ = SHIPPED[name]
    backbone = build()
    offered: list[Any] = [
        backbone.native_head((stream,), width, 3) for stream, width in backbone.feature_dims().items()
    ]
    if pyramid := backbone.pyramid():
        offered.append(backbone.native_head(pyramid, tuple(backbone.feature_dim(level) for level in pyramid), 3))

    assert all(head is None or isinstance(head, nn.Module) for head in offered)
