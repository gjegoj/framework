"""One stream a backbone published, brought to the width a run declared."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor, nn

from src.core import Axis, TensorShape
from src.models.base import Neck
from src.models.registry import neck_registry


@neck_registry.register("projector")
class Projector(Neck):
    """One of the backbone's streams through a single linear layer, at a width the run writes down.

    What it is for: two networks meeting in one space. A teacher and a student publish features of
    whatever width their libraries chose — 1024 and 1280 — and nothing about the two can be compared.
    A projector on either brings them to one declared width, and from there an objective over the
    features has two tensors of the same shape, and a head trained on one of them reads the other.

    One linear layer, and no knob for a second: a stack of projections is what a head is, and ``mlp``
    is where a run declares one. Two places building that stack would be one idea with two homes.

    The brought stream keeps its name, so that nothing downstream moves: a head reads ``pooled`` as it
    read it, a kind's default stream still resolves, and a teacher's declaration is the student's with
    one line changed. Streams this one does not bring are published exactly as they arrived.

    Parameters:
        backbone_shapes: What the backbone publishes, which is what this is built from; the framework
            hands it over, since there is no such thing as a neck that does not know what it reads.
        width: How many numbers the brought stream publishes.
        stream: Which stream is brought. Left out where the backbone publishes one, which is the common
            case; named where it publishes several, since there is a real choice then.
    """

    def __init__(self, backbone_shapes: Mapping[str, TensorShape], width: int, stream: str | None = None) -> None:
        super().__init__()
        if width < 1:
            # Torch does not refuse a zero-wide layer, it warns that initialising one is a no-op and
            # builds it, so a run declaring nothing here would train and report on an empty tensor.
            raise ValueError(f"`width` is how many numbers this stream publishes, so it is at least one; got {width}.")
        self.backbone_shapes = dict(backbone_shapes)
        self.width = width
        self.stream = _the_one_brought(self.backbone_shapes, stream)
        self.projection = nn.Linear(_pooled_width(self.backbone_shapes[self.stream], self.stream), width)

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """What the backbone published, with the one brought stream at the declared width."""
        return {**self.backbone_shapes, self.stream: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,))}

    def forward(self, features: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return {**features, self.stream: cast("Tensor", self.projection(features[self.stream]))}


def _the_one_brought(backbone_shapes: Mapping[str, TensorShape], declared: str | None) -> str:
    """Which stream is brought to the width: the one written, or the only one there was to write.

    Derived where a backbone publishes one, because there is no choice to declare then and a run that
    wrote it would be writing down what the declaration already says. Refused where it publishes
    several: picking one would be this neck deciding which half of a network a run meant.

    Named by the position rather than by the family that filled it — `model.backbone` — because a neck
    is built from shapes and never sees the class that published them, which is the whole of why it is
    a neck: what it reads is features, and features carry no library.
    """
    if declared is not None:
        if declared not in backbone_shapes:
            raise ValueError(f"`stream` names {declared!r}, and `model.backbone` publishes {_listed(backbone_shapes)}.")
        return declared
    if len(backbone_shapes) != 1:
        raise ValueError(
            f"`model.backbone` publishes {_listed(backbone_shapes)}, and a projector brings one of them to a "
            f"declared width; `stream` names which of them."
        )
    return next(iter(backbone_shapes))


def _pooled_width(shape: TensorShape, stream: str) -> int:
    """How many numbers the brought stream reads, which is what the layer is sized from.

    The same question ``models.build._width`` asks on behalf of a head, asked here on behalf of the
    projection, because both are sized by that one number and neither can be built without it.
    """
    if shape.axes != (Axis.CHANNELS,):
        raise ValueError(
            f"{stream!r} is [{', '.join(shape.axes)}], and a projector brings a pooled [channels] stream to "
            f"a declared width. A stream that is still spatial is brought to one by a convolution, which "
            f"this neck does not build."
        )
    width = shape.size(Axis.CHANNELS)
    if width is None:
        raise ValueError(f"`model.backbone` declares no width for {stream!r}; a projection cannot be sized.")
    return width


def _listed(backbone_shapes: Mapping[str, TensorShape]) -> str:
    """The streams a backbone publishes, as a message that names them reads."""
    return ", ".join(sorted(backbone_shapes)) or "nothing"
