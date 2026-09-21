"""A backbone publishes what it wraps, with one stream brought to the width a run declared."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Axis, Stream, TensorShape, TensorTree
from src.models.backbones.projector import ProjectorBackbone
from src.models.base import Backbone
from tests.unit.models.conftest import MAP, MAP_WIDTH, NARROW_WIDTH, SIDE, Encoder, Sentences

WIDTH = 3


class Unmeasured(Backbone):
    """A family naming a stream and not its width, which `TensorShape` permits and a projection cannot read.

    Beside the doubles in `conftest`, because those two publish widths and this test is about the one
    that does not: `models.build._width` refuses a head over such a stream, and a projection is sized
    from the same number.
    """

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(None,))}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        raise NotImplementedError


def sentences() -> dict[str, torch.Tensor]:
    """Two rows whose numbers differ, so a projection of them is observable."""
    return {"text": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}


def test_the_stream_it_projects_is_published_at_the_width_the_run_declared() -> None:
    """A head is sized from this shape, so the shape is what makes the declared width take effect."""
    projected = ProjectorBackbone(Sentences(), width=WIDTH)

    assert projected.feature_shapes == {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(WIDTH,))}
    assert tuple(projected(sentences())[Stream.POOLED].shape) == (2, WIDTH)


def test_a_stream_it_does_not_project_is_published_as_the_wrapped_backbone_published_it(
    images: dict[str, torch.Tensor],
) -> None:
    """A run reading a feature map beside a pooled vector goes on reading the map it always read."""
    projected = ProjectorBackbone(Encoder(), width=WIDTH, stream=Stream.POOLED)

    assert projected.feature_shapes[Stream.DECODER] == MAP
    assert tuple(projected(images)[Stream.DECODER].shape) == (2, MAP_WIDTH, SIDE, SIDE)


@pytest.mark.parametrize(
    ("wrapped", "declared", "refused_with"),
    [
        pytest.param(Sentences, {"width": 0}, "at least one", id="a width of none"),
        pytest.param(Encoder, {"width": WIDTH}, "which of them", id="several streams, none named"),
        pytest.param(
            Sentences, {"width": WIDTH, "stream": "encoder"}, "publishes pooled", id="a stream that is not there"
        ),
        pytest.param(Encoder, {"width": WIDTH, "stream": Stream.DECODER}, "still spatial", id="a stream that is a map"),
        pytest.param(Unmeasured, {"width": WIDTH}, "declares no width", id="a stream of no declared width"),
    ],
)
def test_a_projector_that_could_not_publish_a_width_is_refused_where_it_is_declared(
    wrapped: type[Backbone], declared: dict[str, Any], refused_with: str
) -> None:
    """Each of these otherwise dies on the first matmul of a run, a thousand steps in and naming nothing."""
    with pytest.raises(ValueError, match=refused_with):
        ProjectorBackbone(wrapped(), **declared)


def test_a_network_that_publishes_no_streams_is_refused_where_it_is_declared() -> None:
    """This one brings a stream of another to a width, so what it wraps has to publish streams."""
    with pytest.raises(TypeError, match="Backbone"):
        ProjectorBackbone(nn.Linear(2, 2), width=WIDTH)  # type: ignore[arg-type]


def test_the_classifier_the_wrapped_file_carried_is_not_carried_on_and_the_run_is_told(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Those rows were read off the features this backbone replaced, so no head here could hold them.

    Said rather than refused: the file is named for the encoder's weights and they arrive either way,
    but a warm start a run asked for and silently did not get is the defect this says out loud.
    """
    carrying = Sentences()
    carrying.carried_head = {"fc.weight": torch.zeros(2, NARROW_WIDTH)}

    with caplog.at_level(logging.INFO):
        projected = ProjectorBackbone(carrying, width=WIDTH)

    said = "\n".join(caplog.messages)
    assert projected.carried_head == {}
    assert "starts fresh" in said
    assert f"reads {NARROW_WIDTH} features" in said, said
    assert f"publishes {WIDTH}" in said, said


def test_it_offers_no_head_of_its_own_though_the_family_it_wraps_does() -> None:
    """`multiview` passes the wrapped family's classifier on, because drawing views changes nothing a
    head reads; bringing a stream to a width changes exactly that, so the same classifier would be a
    head of the width that is gone, and a run asking for `native` is told so where heads are built."""
    wrapped = Encoder()
    projected = ProjectorBackbone(wrapped, width=WIDTH, stream=Stream.POOLED)

    assert wrapped.native_head(Stream.POOLED, 2) is not None
    assert projected.native_head(Stream.POOLED, 2) is None
