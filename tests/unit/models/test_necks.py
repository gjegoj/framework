"""A neck reads the streams a backbone published and publishes its own."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Axis, Stream, TensorShape
from src.models.necks.projector import Projector
from tests.unit.models.conftest import MAP, MAP_WIDTH, NARROW, NARROW_WIDTH, POOLED_WIDTH, SIDE, VECTOR

WIDTH = 3


def test_the_stream_it_brings_is_published_at_the_width_the_run_declared() -> None:
    """A head is sized from this shape, so the shape is what makes the declared width take effect."""
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH)

    assert brought.feature_shapes == {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(WIDTH,))}
    assert tuple(brought({Stream.POOLED: torch.ones(2, NARROW_WIDTH)})[Stream.POOLED].shape) == (2, WIDTH)


def test_a_stream_it_does_not_bring_is_published_as_the_backbone_published_it() -> None:
    """A run reading a feature map beside a pooled vector goes on reading the map it always read."""
    brought = Projector(backbone_shapes={Stream.POOLED: VECTOR, Stream.DECODER: MAP}, width=WIDTH, stream=Stream.POOLED)
    passing = {
        Stream.POOLED: torch.ones(2, POOLED_WIDTH),
        Stream.DECODER: torch.ones(2, MAP_WIDTH, SIDE, SIDE),
    }

    assert brought.feature_shapes[Stream.DECODER] == MAP
    assert tuple(brought(passing)[Stream.DECODER].shape) == (2, MAP_WIDTH, SIDE, SIDE)


@pytest.mark.parametrize(
    ("declared", "refused_with"),
    [
        pytest.param({"backbone_shapes": {Stream.POOLED: NARROW}, "width": 0}, "at least one", id="a width of none"),
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: VECTOR, Stream.DECODER: MAP}, "width": WIDTH},
            "which of them",
            id="several streams, none named",
        ),
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: NARROW}, "width": WIDTH, "stream": "encoder"},
            "publishes pooled",
            id="a stream that is not there",
        ),
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: VECTOR, Stream.DECODER: MAP}, "width": WIDTH, "stream": Stream.DECODER},
            "still spatial",
            id="a stream that is a map",
        ),
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(None,))}, "width": WIDTH},
            "declares no width",
            id="a stream of no declared width",
        ),
    ],
)
def test_a_neck_that_could_not_publish_a_width_is_refused_where_it_is_declared(
    declared: dict[str, Any], refused_with: str
) -> None:
    """Each of these otherwise dies on the first matmul of a run, a thousand steps in and naming nothing."""
    with pytest.raises(ValueError, match=refused_with):
        Projector(**declared)
