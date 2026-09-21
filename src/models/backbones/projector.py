"""A backbone that publishes what it wraps, with one stream brought to a width the run declared."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

from torch import Tensor, nn

from src.core import Axis, TensorShape, TensorTree
from src.models.base import Backbone
from src.models.registry import backbone_registry

log = logging.getLogger(__name__)


@backbone_registry.register("projector")
class ProjectorBackbone(Backbone):
    """What its backbone publishes, with one stream brought to the width a run declared.

    Declared around a backbone rather than instead of one, the way ``multiview`` is, so every family
    stays available to a run that needs a width of its own and none of them learns what a projection is.

    What it is for: two networks meeting in one space. A teacher and a student publish features of
    whatever width their libraries chose — 1024 and 1280 — and nothing about the two can be compared.
    A projector on either brings them to one declared width, and from there an objective over the
    features has two tensors of the same shape, and a head trained on one of them reads the other.

    One linear layer, and no knob for a second: a stack of projections is what a head is, and ``mlp``
    is where a run declares one. Two places building that stack would be one idea with two homes.

    The brought stream keeps its name, so that nothing downstream moves: a head reads ``pooled`` as it
    read it, a kind's default stream still resolves, and a teacher's declaration is the student's with
    one word changed. Streams this one does not bring are published exactly as they arrived.

    This family offers no head of its own. The wrapped library's classifier is sized for the features
    this one replaced, so ``native_head`` stays the base's ``None`` and a run declaring ``head: native``
    is refused by name where heads are built, rather than handed a classifier of the wrong width.

    Parameters:
        backbone: The network whose features are brought to a width; any family, declared by ``_target_``.
        width: How many numbers the brought stream publishes.
        stream: Which stream is brought. Left out where the backbone publishes one, which is the common
            case; named where it publishes several, since there is a real choice then.
    """

    def __init__(self, backbone: Backbone, width: int, stream: str | None = None) -> None:
        super().__init__()
        if not isinstance(backbone, Backbone):
            raise TypeError(
                f"{type(backbone).__name__} is not a Backbone: this one brings a stream of another to a "
                "declared width, so it needs a network that names its feature streams."
            )
        if width < 1:
            raise ValueError(f"`width` is how many numbers this stream publishes, so it is at least one; got {width}.")
        published = backbone.feature_shapes
        family = type(backbone).__name__
        self.stream = _the_one_brought(published, stream, family)
        self.width = width
        self.backbone = backbone
        self.projection = nn.Linear(_pooled_width(published[self.stream], self.stream, family), width)
        if backbone.carried_head:
            # Not carried on, and not in silence. Those rows were read off the features this backbone
            # has replaced, so no head built over it could hold them; `multiview` carries them because
            # drawing views changes nothing a head reads, and bringing a stream to a width changes
            # exactly that. Said rather than refused: the file is named for the encoder's weights, and
            # those arrive either way.
            log.info(
                "The classifier the file %s started from carried reads %d features, and %r now publishes "
                "%d, so its rows have nowhere to land and the head over it starts fresh.",
                family,
                self.projection.in_features,
                self.stream,
                width,
            )

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """What the wrapped backbone publishes, with the one brought stream at the declared width."""
        return {
            **self.backbone.feature_shapes,
            self.stream: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,)),
        }

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        encoded = self.backbone(inputs)
        return {**encoded, self.stream: cast("Tensor", self.projection(encoded[self.stream]))}


def _the_one_brought(published: Mapping[str, TensorShape], declared: str | None, family: str) -> str:
    """Which stream is brought to the width: the one written, or the only one there was to write.

    Derived where a backbone publishes one, because there is no choice to declare then and a run that
    wrote it would be writing down what the declaration already says. Refused where it publishes
    several: picking one would be this family deciding which half of a network a run meant.
    """
    if declared is not None:
        if declared not in published:
            raise ValueError(f"`stream` names {declared!r}, and {family} publishes {_listed(published)}.")
        return declared
    if len(published) != 1:
        raise ValueError(
            f"{family} publishes {_listed(published)}, and a projector brings one of them to a declared "
            f"width; `stream` names which of them."
        )
    return next(iter(published))


def _pooled_width(shape: TensorShape, stream: str, family: str) -> int:
    """How many numbers the brought stream reads, which is what the layer is sized from.

    The same question ``models.build._width`` asks on behalf of a head, asked here on behalf of the
    projection, because both are sized by that one number and neither can be built without it.
    """
    if shape.axes != (Axis.CHANNELS,):
        raise ValueError(
            f"{stream!r} is [{', '.join(shape.axes)}], and a projector brings a pooled [channels] stream to "
            f"a declared width. A stream that is still spatial is brought to one by a convolution, which "
            f"this family does not build."
        )
    width = shape.size(Axis.CHANNELS)
    if width is None:
        raise ValueError(f"{family} declares no width for {stream!r}; a projection cannot be sized.")
    return width


def _listed(published: Mapping[str, TensorShape]) -> str:
    """The streams a family publishes, as a message that names them reads."""
    return ", ".join(sorted(published)) or "nothing"
