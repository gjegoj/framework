"""One backbone, one head per task: the graph publishes both what heads produced and what they read."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor, nn

from src.core import ModelOutput, Representation, Stream, require_tensor
from src.models import CompositeModel, HeadConnection, Model
from src.models.heads import CosineHead, ExpandedHead, StackedHeads
from src.models.necks.projector import Projector
from tests.unit.models.conftest import MAP_WIDTH, POOLED_WIDTH, SIDE, CompositeFactory, Encoder

CLASSES = 2


def grown() -> nn.Module:
    """A class space that grew: the declared head over the rows a file carried, beside the fresh ones."""
    return ExpandedHead(base=CosineHead(POOLED_WIDTH, 1), novel=CosineHead(POOLED_WIDTH, 1))


def paired() -> nn.Module:
    """A pairing: the declared head built once per stream of it."""
    return StackedHeads({"first": CosineHead(POOLED_WIDTH, CLASSES), "second": CosineHead(POOLED_WIDTH, CLASSES)})


@pytest.mark.parametrize("wrapped", [grown, paired], ids=["a class space that grew", "a pairing"])
def test_a_head_this_framework_wrapped_still_answers_for_what_it_wraps(
    make_composite: CompositeFactory, wrapped: Callable[[], nn.Module]
) -> None:
    """Both wrappers are one declaration built more than once, so what it answers with is what they do.

    Read by the composition root, which refuses an objective that reads something else. Answering
    `projected` about a tensor of cosines made that refusal tell a run to declare the head it had
    already declared, and would have let cross-entropy softmax angles and publish them as confidence.
    """
    model = make_composite(heads={"label": HeadConnection(wrapped(), streams=(Stream.POOLED,))})

    assert model.produces("label") is Representation.COSINES


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
            "label": HeadConnection(nn.Linear(POOLED_WIDTH, 2), streams=(Stream.POOLED,)),
            "mask": HeadConnection(nn.Conv2d(MAP_WIDTH, 3, 1), streams=(Stream.DECODER,)),
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


def test_each_task_owns_its_head_and_shares_the_backbone(make_composite: CompositeFactory) -> None:
    """What a run may give a task its own learning rate over: its head, and nothing of the encoder."""
    model = make_composite()

    owned = {id(parameter) for parameter in model.parameters_of("label")}

    assert owned == {id(parameter) for parameter in model.heads["label"].parameters()}
    assert owned.isdisjoint(id(parameter) for parameter in model.backbone.parameters())
    assert list(model.parameters_of("absent")) == []


def test_a_neck_stands_between_the_backbone_and_the_heads_that_read_it(
    backbone: Encoder, images: dict[str, Tensor]
) -> None:
    """Both halves of what a neck being a position means: the head reads what the neck published, and
    what the model publishes as its features is what the neck published too.

    The second half is what a term of `learner.loss` naming a stream compares, so a neck whose output
    reached the head and not the report would have two networks pulled towards features neither of
    them answers through.
    """
    brought = Projector(backbone_shapes=backbone.feature_shapes, width=3, stream=Stream.POOLED)
    model = CompositeModel(
        backbone,
        {"label": HeadConnection(nn.Linear(3, CLASSES), streams=(Stream.POOLED,))},
        neck=brought,
    )

    output = model(images)

    assert require_tensor(output.outputs["label"], name="label").shape == (2, CLASSES)
    assert require_tensor(output.features[Stream.POOLED], name=Stream.POOLED).shape == (2, 3)
    assert require_tensor(output.features[Stream.DECODER], name=Stream.DECODER).shape == (2, MAP_WIDTH, SIDE, SIDE)
