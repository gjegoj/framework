"""One stream a backbone published, brought to the width a run declared."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import cast, override

from torch import Tensor, nn
from torch.nn.functional import normalize

from src.core import FEATURE_AXIS, Axis, TensorShape
from src.models.base import Neck, PublishesStreams
from src.models.heads import Mlp
from src.models.registry import neck_registry


@neck_registry.register("projector")
class Projector(Neck):
    """One of the backbone's streams brought to a width the run writes down.

    What it is for: two networks meeting in one space. A teacher and a student publish features of
    whatever width their libraries chose — 1024 and 1280 — and nothing about the two can be compared.
    A projector on either brings them to one declared width, and from there an objective over the
    features has two tensors of the same shape, and a head trained on one of them reads the other.

    Hidden widths make it the same stack ``mlp`` is, built by ``Mlp`` itself rather than beside it, so
    that stack still has one home — a head is any module built at two widths, and a stack of them is
    what this position needs. The word is the head's word and means what it means there; only the
    default differs, and it differs by what each name promises. An ``mlp`` without hidden widths would
    not be one, so it takes the one its name implies; a ``projector`` without them is the single
    projection its name says. Declared empty it is refused either way, being a second spelling of a
    stack that is already declared by leaving the word out.

    ``width`` rather than ``out_features``, though it is the number ``Mlp`` is built at: measured,
    ``out_features`` is a word a declaration may not write — ``refuse_restated_facts`` rejects a head
    that states it, since the framework derives it from the task. Here it is the run's own free choice,
    and one word would read as "the framework fills this in" on one line of ``model:`` and "you must
    write this" on the next.

    ``norm`` puts a normalization at the end of this neck, which is both where the heads read from and
    where a term of ``learner.loss`` compares. Matching a stream that carries a scale spends the
    distance on a magnitude the head cannot see; and a head trained on normalized features reads a
    space it never saw if the network it was transplanted onto does not normalize — which is why a
    teacher is trained *with* its norm rather than handed one afterwards. Neither kind carries a
    parameter, so nothing has to be frozen or transplanted with them, and both publish at the same
    length. They are not each other: measured on a stream with a common component, the two answers sit
    at cos 0.894.

    The brought stream keeps its name, so that nothing downstream moves: a head reads ``pooled`` as it
    read it, a kind's default stream still resolves, and a teacher's declaration is the student's with
    one line changed. Streams this one does not bring are published exactly as they arrived.

    Parameters:
        backbone_shapes: What the backbone publishes, which is what this is built from; the framework
            hands it over, since there is no such thing as a neck that does not know what it reads.
        width: How many numbers the brought stream publishes.
        stream: Which stream is brought. Left out where the backbone publishes one, which is the common
            case; named where it publishes several, since there is a real choice then.
        hidden_features: The width of each layer between what is read and what is published, in the
            order they are read through — the same word ``mlp`` writes, for the same thing. Left out,
            there are none, and this is the one projection its name says.
        norm: How the brought stream is normalized before the heads read it, or nothing where a run
            wants the projection as it is. Whatever this run writes, the network it distils from was
            trained with — a norm is not something a trained head can be given afterwards.
    """

    def __init__(
        self,
        backbone_shapes: Mapping[str, TensorShape],
        width: int,
        stream: str | None = None,
        hidden_features: Sequence[int] | None = None,
        norm: str | None = None,
    ) -> None:
        super().__init__()
        if width < 1:
            # Torch does not refuse a zero-wide layer, it warns that initialising one is a no-op and
            # builds it, so a run declaring nothing here would train and report on an empty tensor.
            raise ValueError(f"`width` is how many numbers this stream publishes, so it is at least one; got {width}.")
        if hidden_features is not None and not hidden_features:
            raise ValueError(
                "`hidden_features` is the widths between what this neck reads and what it publishes, and an "
                "empty list declares none of them; a projector of one projection is what leaving it out gives."
            )
        self.backbone_shapes = dict(backbone_shapes)
        self.width = width
        self.stream = _the_one_brought(self.backbone_shapes, stream)
        reads = _pooled_width(self.backbone_shapes[self.stream], self.stream)
        self.projection = (
            nn.Linear(reads, width)
            if hidden_features is None
            else Mlp(in_features=reads, out_features=width, hidden_features=hidden_features)
        )
        self.norm = _brought_to_one_scale(norm, width)

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """What the backbone published, with the one brought stream at the declared width."""
        return {**self.backbone_shapes, self.stream: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,))}

    def forward(self, features: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        return self.forward_intermediates(features)[0]

    @override
    def forward_intermediates(
        self, features: Mapping[str, Tensor], /
    ) -> tuple[Mapping[str, Tensor], Mapping[str, Tensor]]:
        """The brought stream, and every width the stack passed through on the way to it.

        Asked of the stack rather than walked here, because the stack *is* ``Mlp`` wherever a run
        declared hidden widths — the same class a head is, already publishing under the same names in
        the same order. Which widths a stack publishes has one home, and it is not this one. A single
        ``Linear`` is not that class and holds nothing between, so it publishes nothing and says so.

        What comes back is the projection before the norm, because the norm stands on the stack's
        answer rather than inside it: a term comparing a neck's middle compares the scale along with
        the direction, and what that is worth is the term's weight to say.

        ``forward`` is this method's first element rather than a second pass written beside it, so the
        two cannot drift apart — the arrangement ``Mlp`` keeps for the same reason.
        """
        read = features[self.stream]
        projected, published = (
            self.projection.forward_intermediates(read)
            if isinstance(self.projection, PublishesStreams)
            else (cast("Tensor", self.projection(read)), {})
        )
        return {**features, self.stream: cast("Tensor", self.norm(projected))}, published


class L2Norm(nn.Module):
    """The stream read as a direction alone, at the length a unit-variance vector of this width has.

    ``sqrt(width)`` rather than one, and fixed rather than learned. Fixed, because a learned scale on
    the student would be one number the frozen head depends on and nothing could hold still — ``freeze``
    addresses modules by path, and a bare parameter has no path to write. ``sqrt(width)``, because it is
    the length ``layer_norm`` publishes at, so which of the two a run declares does not change what a
    term's weight means: measured at 512, mean row norm 22.627 either way.

    Not in the neck registry: nobody declares one. It is what ``norm: l2`` builds.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale: float = width**0.5

    def forward(self, features: Tensor) -> Tensor:
        return normalize(features, dim=FEATURE_AXIS) * self.scale


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


NORMALIZATIONS: Mapping[str, Callable[[int], nn.Module]] = {
    # No affine on the layer norm: the scale and the shift are the two things the head after it is meant
    # not to see, and learning them back is handing them to it again.
    "layer_norm": partial(nn.LayerNorm, elementwise_affine=False),
    "l2": L2Norm,
}
"""What ``norm`` may name, and what each of those words builds, in one statement.

A mapping rather than an enum beside a branch, because those are two statements about one closed set:
a word added to one and forgotten in the other falls through to whichever layer the branch ends on,
and a run gets a normalization it did not write. Here a word nobody built is not a word.
"""


def _brought_to_one_scale(declared: str | None, width: int) -> nn.Module:
    """The normalization a run wrote at the end of its neck, or nothing where it wrote none.

    Checked here, rather than trusted, because this arrives from a config through ``instantiate``,
    which is untyped passthrough: a declaration nothing typed has no owner but whoever consumes it.
    """
    if declared is None:
        return nn.Identity()
    if declared not in NORMALIZATIONS:
        raise ValueError(
            f"`norm` names {declared!r}, and a projector normalizes with {', '.join(sorted(NORMALIZATIONS))}. "
            f"Left out, it publishes the projection as it is."
        )
    return NORMALIZATIONS[declared](width)


def _listed(backbone_shapes: Mapping[str, TensorShape]) -> str:
    """The streams a backbone publishes, as a message that names them reads."""
    return ", ".join(sorted(backbone_shapes)) or "nothing"
