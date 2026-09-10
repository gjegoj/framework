"""One backbone, one head per task: the graph publishes both what heads produced and what they read."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.core import ModelOutput, Stream, require_tensor
from src.models import CompositeModel, HeadConnection, Model
from tests.models.conftest import MAP_WIDTH, POOLED_WIDTH, SIDE, CompositeFactory, Encoder


def test_a_composite_is_a_model_and_registers_its_parts_where_config_addresses_them(
    make_composite: CompositeFactory,
) -> None:
    """`backbone` and `heads.<task>` are the paths a freeze callback and a checkpoint name."""
    model = make_composite()

    assert isinstance(model, Model)
    assert {name for name, _ in model.named_children()} == {"backbone", "heads"}
    assert "label" in dict(model.heads.named_children())


def test_encodes_once_and_serves_every_task_from_the_stream_it_declared(
    backbone: Encoder, images: dict[str, Tensor]
) -> None:
    model = CompositeModel(
        backbone,
        {
            "label": HeadConnection(nn.Linear(POOLED_WIDTH, 2), input=Stream.POOLED),
            "mask": HeadConnection(nn.Conv2d(MAP_WIDTH, 3, 1), input=Stream.DECODER),
        },
    )

    output = model(images)

    assert isinstance(output, ModelOutput)
    assert require_tensor(output.outputs["label"], name="label").shape == (2, 2)
    assert require_tensor(output.outputs["mask"], name="mask").shape == (2, 3, SIDE, SIDE)
    assert set(output.features) == {Stream.POOLED, Stream.DECODER}


def test_the_gradient_of_every_head_reaches_the_shared_backbone(
    make_composite: CompositeFactory, images: dict[str, Tensor], backbone: Encoder
) -> None:
    """One encoding shared by every task is the point of the family; a detached head would train nothing."""
    model = make_composite()

    require_tensor(model(images).outputs["label"], name="label").sum().backward()

    assert backbone.projection.weight.grad is not None and torch.any(backbone.projection.weight.grad != 0)


@pytest.mark.parametrize(
    ("heads", "reason"),
    [
        pytest.param({}, "at least one", id="no heads"),
        pytest.param({"a/b": None}, "names must be", id="a name that cannot be a metric key"),
        pytest.param({"label": "logits"}, "pooled, decoder", id="a stream the backbone does not publish"),
    ],
)
def test_refuses_a_graph_that_could_not_run(backbone: Encoder, heads: dict[str, str], reason: str) -> None:
    connections = {name: HeadConnection(nn.Identity(), input=stream or Stream.POOLED) for name, stream in heads.items()}

    with pytest.raises(ValueError, match=reason):
        CompositeModel(backbone, connections)


def test_a_head_reads_one_named_stream(backbone: Encoder) -> None:
    with pytest.raises(ValueError, match="nonblank"):
        HeadConnection(nn.Identity(), input=" ")
