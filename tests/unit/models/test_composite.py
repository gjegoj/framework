"""One backbone, one head per task: the graph publishes both what heads produced and what they read."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
import torch
from torch import Tensor, nn

from src.core import ModelOutput, Representation, Stream, TensorTree, require_tensor
from src.models import CompositeModel, HeadConnection, Model
from src.models.heads import CosineHead, ExpandedHead, Mlp, StackedHeads
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


def test_what_a_neck_publishes_on_its_way_is_reported_under_the_position_it_occupies(
    backbone: Encoder, images: dict[str, Tensor]
) -> None:
    """A neck has no task to be named after and no name of its own — it is a position, and this model is
    what registers it there. Filed beside the heads' streams by the same rule, so a term of
    `learner.loss` names one the way it names the other.

    The level a distilled pair shares that neither backbone offers: two necks declared alike publish
    the same width between what they read and what they answer, while two families' own middles are
    not comparable at all.
    """
    brought = Projector(backbone_shapes=backbone.feature_shapes, width=3, stream=Stream.POOLED, hidden_features=[4])
    model = CompositeModel(
        backbone,
        {"label": HeadConnection(Mlp(3, CLASSES, hidden_features=[5]), streams=(Stream.POOLED,))},
        neck=brought,
    )

    output = model(images)

    assert set(output.features) == {Stream.POOLED, Stream.DECODER, "neck_hidden_0", "label_hidden_0"}
    assert require_tensor(output.features["neck_hidden_0"], name="neck_hidden_0").shape == (2, 4)
    assert require_tensor(output.features[Stream.POOLED], name=Stream.POOLED).shape == (2, 3)


def test_a_task_named_after_the_neck_position_is_refused_where_both_are_assembled(backbone: Encoder) -> None:
    """Both would file what they publish under `neck_<stream>`, and a mapping lets one answer for the
    other without a word. Refused by whoever can see both at once, which is neither of them."""
    brought = Projector(backbone_shapes=backbone.feature_shapes, width=3, stream=Stream.POOLED, hidden_features=[4])

    with pytest.raises(ValueError, match="Rename the task"):
        CompositeModel(
            backbone,
            {"neck": HeadConnection(Mlp(3, CLASSES, hidden_features=[5]), streams=(Stream.POOLED,))},
            neck=brought,
        )


def test_what_a_head_publishes_on_its_way_to_an_answer_is_reported_under_the_task_it_answers(
    backbone: Encoder, images: dict[str, Tensor]
) -> None:
    """A term of `learner.loss` names a stream, and two tasks may declare the same head — so a head's
    stream is filed under the task, by the composite, which alone knows what the head was registered as.

    The answer is untouched: what the run is judged by does not change because more of the network has
    become visible to a term.
    """
    model = CompositeModel(
        backbone,
        {"label": HeadConnection(Mlp(POOLED_WIDTH, CLASSES, hidden_features=[5, 4]), streams=(Stream.POOLED,))},
    )

    output = model(images)

    head = model.heads["label"]
    assert isinstance(head, Mlp)
    assert set(output.features) == {Stream.POOLED, Stream.DECODER, "label_hidden_0", "label_hidden_1"}
    assert require_tensor(output.features["label_hidden_0"], name="label_hidden_0").shape == (2, 5)
    assert torch.equal(
        require_tensor(output.features["label_hidden_0"], name="label_hidden_0"),
        head.layers[0](require_tensor(output.features[Stream.POOLED], name=Stream.POOLED)),
    )
    assert require_tensor(output.outputs["label"], name="label").shape == (2, CLASSES)


def test_a_head_with_nothing_to_publish_leaves_the_features_exactly_what_the_encoding_half_published(
    make_composite: CompositeFactory, images: dict[str, Tensor]
) -> None:
    """Every ordinary run: a `linear` head declares no capability, and what the report and a term see is
    the backbone's streams and not one key more — the position costs a run that does not use it nothing.
    """
    output = make_composite()(images)

    assert set(output.features) == {Stream.POOLED, Stream.DECODER}


class Handing(Encoder):
    """An encoder that keeps a reference to the mapping it handed over, which is the only way a caller
    growing that very mapping rather than a copy of it is observable from outside."""

    def __init__(self) -> None:
        super().__init__()
        self.handed: Mapping[str, Tensor] = {}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        self.handed = super().forward(inputs)
        return self.handed


def test_what_a_head_publishes_does_not_grow_the_features_the_heads_were_handed(
    images: dict[str, Tensor],
) -> None:
    """Two mappings for two things, and this is the half that makes the other one structural: a head's
    stream reaches the report without reaching what the heads read, so no head can read another head's
    stream however the tasks are ordered — and nothing the backbone kept is written into behind its back.
    """
    backbone = Handing()
    model = CompositeModel(
        backbone,
        {"label": HeadConnection(Mlp(POOLED_WIDTH, CLASSES, hidden_features=[5]), streams=(Stream.POOLED,))},
    )

    output = model(images)

    assert set(backbone.handed) == {Stream.POOLED, Stream.DECODER}
    assert "label_hidden_0" in output.features
