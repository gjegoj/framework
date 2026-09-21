"""A neck reads the streams a backbone published and publishes its own."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Axis, Stream, TensorShape
from src.models.base import Neck
from src.models.heads import Mlp
from src.models.necks.projector import NORMALIZATIONS, Projector
from src.models.weights import load_weights
from tests.unit.models.conftest import MAP, MAP_WIDTH, NARROW, NARROW_WIDTH, POOLED_WIDTH, SIDE, VECTOR

WIDTH = 3

NORMED = 64
"""The width the tests about `norm` declare, where the rest of this file declares three.

`layer_norm` divides by `sqrt(var + eps)`, so how far its answer sits from a perfectly scale-free one is
set by `eps` against a row's variance rather than by the width — and over three numbers a row's variance
is often small enough to make `eps` a visible share of it. Measured over 200 draws: the loudest
projection moves the answer by 3.5e-02 at three and 1.7e-04 at sixty-four, and the length published
drifts 3.8e-01 at three and 8.5e-05. A neck a run declares is hundreds wide, so sixty-four is the regime
these claims are about, and three would make them pass or fail on the draw.
"""


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


def test_declared_hidden_widths_are_the_stack_a_head_builds_between_what_is_read_and_what_is_published() -> None:
    """A neck is where two networks are brought to one space, and that space is the teacher's own tail:
    one layer fewer, or one width off, and what a term compares is another space entirely."""
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH, hidden_features=[7])

    widths = [(one.in_features, one.out_features) for one in brought.modules() if isinstance(one, nn.Linear)]

    assert widths == [(NARROW_WIDTH, 7), (7, WIDTH)]


def test_a_projector_of_one_stack_does_not_take_the_weights_of_another() -> None:
    """What keeps a run from continuing from a neck of another shape in silence: the two stacks are told
    apart by the names they register, so a file of one is refused by name rather than half-loaded."""
    one = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH)
    stacked = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH, hidden_features=[7])

    with pytest.raises(ValueError, match="does not fit"):
        load_weights(stacked, one.state_dict(), "narrowed.ckpt")


@pytest.mark.parametrize("norm", list(NORMALIZATIONS))
def test_a_norm_leaves_the_heads_reading_the_same_thing_however_loud_the_projection_is(norm: str) -> None:
    """The whole of what the position buys. Under a distance toward a target it cannot reach a student
    publishes features shrunk toward their mean, and a head trained on features of full variance then
    reads a distribution that is not its own; with a norm before it, only the direction reaches it.

    Tolerated to 1e-3 rather than tighter because `layer_norm` is scale-free only up to its `eps`:
    measured over 200 draws at this width, the loudest projection moves the answer by 1.7e-04, and `l2`
    — which is exactly scale-free — by 7.2e-07.
    """
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=NORMED, norm=norm)
    features = {Stream.POOLED: torch.randn(4, NARROW_WIDTH)}
    quiet = brought(features)[Stream.POOLED]

    with torch.no_grad():
        for parameter in brought.projection.parameters():
            parameter.mul_(100)

    assert torch.allclose(quiet, brought(features)[Stream.POOLED], atol=1e-3)


@pytest.mark.parametrize("norm", list(NORMALIZATIONS))
def test_either_norm_publishes_at_one_length_so_a_weight_set_for_one_means_the_same_for_the_other(
    norm: str,
) -> None:
    """Why a run may swap them without touching `learner.loss`: `sqrt(width)` is the length `layer_norm`
    publishes at — it divides by the spread it just measured — so `l2` is given that same length rather
    than one. Measured at 512, mean row norm 22.627 either way.

    Relative rather than absolute, and measured over 200 draws at this width: `layer_norm` lands within
    8.5e-05 of `sqrt(width)` and `l2` within 1.8e-07, being given the length rather than arriving at it.
    """
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=NORMED, norm=norm)

    published = brought({Stream.POOLED: torch.randn(64, NARROW_WIDTH)})[Stream.POOLED]

    assert torch.allclose(published.norm(dim=-1), torch.full((64,), NORMED**0.5), rtol=1e-3, atol=0.0)


def test_layer_norm_takes_the_common_component_that_a_direction_alone_keeps() -> None:
    """Which of them a run declares is a real choice, so they have to be really different: measured on a
    stream carrying a shared direction — which is what a backbone's features carry — the two answer at
    cos 0.894. `layer_norm` removes the mean over the width as well as the scale, `l2` only the scale.
    """
    published: dict[str, Tensor] = {}
    for norm in NORMALIZATIONS:
        neck = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=NORMED, norm=norm)
        assert isinstance(neck.projection, nn.Linear)
        with torch.no_grad():
            # A common component a few times the spread beneath it, which is what a backbone publishes:
            # measured, this leaves a pre-norm row of std 0.55 around a mean of 2.
            neck.projection.bias.fill_(2.0)
        published[norm] = neck({Stream.POOLED: torch.randn(64, NARROW_WIDTH)})[Stream.POOLED]

    assert bool(published["layer_norm"].mean(-1).abs().max() < 1e-5)
    assert bool(published["l2"].mean(-1).abs().min() > 1e-2)


def test_a_stream_brought_without_a_norm_is_published_exactly_as_it_was_projected() -> None:
    """The knob is a knob: a run that does not write it gets the projector it had before the knob
    existed, down to the tensor, so nothing already written starts reporting different numbers."""
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH)
    features = torch.randn(4, NARROW_WIDTH)

    assert torch.equal(brought({Stream.POOLED: features})[Stream.POOLED], brought.projection(features))


def test_a_neck_with_nothing_between_answers_with_its_streams_and_publishes_none() -> None:
    """The default every neck inherits, and what a run declaring one of its own gets without saying so:
    a neck with nothing between what it reads and what it publishes answers that it published none,
    rather than leaving whoever assembles it to ask whether it may be asked."""

    class Passing(Neck):
        @property
        def feature_shapes(self) -> Mapping[str, TensorShape]:
            return {Stream.POOLED: NARROW}

        def forward(self, features: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
            return features

    read: dict[str, Tensor] = {Stream.POOLED: torch.randn(2, NARROW_WIDTH)}

    answered, published = Passing().forward_intermediates(read)

    assert answered is read
    assert published == {}


def test_a_neck_declared_with_hidden_widths_publishes_them_as_the_projections_they_are() -> None:
    """What a term reaches at a neck's middle: each width the stack passed through, under the name and
    in the order the stack itself gives it — the same class a head is, so a term names a neck's middle
    exactly as it names a head's."""
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH, hidden_features=[7])

    read: dict[str, Tensor] = {Stream.POOLED: torch.randn(4, NARROW_WIDTH)}

    answered, published = brought.forward_intermediates(read)

    assert list(published) == ["hidden_0"]
    assert tuple(published["hidden_0"].shape) == (4, 7)
    assert tuple(answered[Stream.POOLED].shape) == (4, WIDTH)


def test_the_norm_at_the_end_of_a_neck_stands_on_the_stacks_answer_and_not_inside_it() -> None:
    """Which widths a stack publishes has one home and it is not this neck, so what the stack published
    arrives here untouched and the norm is put on the answer alone. A term over `neck_hidden_0`
    therefore compares a distance carrying scale, where one over the brought stream under a norm
    compares direction alone — and what that is worth is the term's weight to say.

    Against the stack itself and the norm itself rather than against a second neck: two necks of one
    set of weights would answer alike under any change made to both, so a norm that reached into the
    middle of every neck would pass such a test unseen. Exactly rather than within a tolerance, so
    this is not a claim about a draw either.
    """
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH, hidden_features=[7], norm="layer_norm")
    assert isinstance(brought.projection, Mlp)
    read = torch.randn(4, NARROW_WIDTH)

    answered, published = brought.forward_intermediates({Stream.POOLED: read})

    assert torch.equal(published["hidden_0"], brought.projection.forward_intermediates(read)[1]["hidden_0"])
    assert torch.equal(answered[Stream.POOLED], brought.norm(brought.projection(read)))


def test_a_neck_of_one_projection_publishes_nothing_because_it_has_nothing_between() -> None:
    """The knob is the whole of it: a run that declared no hidden widths has no middle to compare at,
    so there is no name for a term to reach and this neck answers that there is none."""
    brought = Projector(backbone_shapes={Stream.POOLED: NARROW}, width=WIDTH)

    read: dict[str, Tensor] = {Stream.POOLED: torch.randn(4, NARROW_WIDTH)}

    assert brought.forward_intermediates(read)[1] == {}


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
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: NARROW}, "width": WIDTH, "hidden_features": []},
            "leaving it out",
            id="a stack of no layers, which is what leaving it out already declares",
        ),
        pytest.param(
            {"backbone_shapes": {Stream.POOLED: NARROW}, "width": WIDTH, "norm": "layernorm"},
            "l2, layer_norm",
            id="a normalization spelled the way it is not written",
        ),
    ],
)
def test_a_neck_that_could_not_publish_a_width_is_refused_where_it_is_declared(
    declared: dict[str, Any], refused_with: str
) -> None:
    """Each of these otherwise dies on the first matmul of a run, a thousand steps in and naming nothing."""
    with pytest.raises(ValueError, match=refused_with):
        Projector(**declared)
