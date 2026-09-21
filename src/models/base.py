"""What a network is to this framework: named inputs in, named outputs out, and the parts a run composes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from torch import Tensor, nn

from src.core import ModelOutput, Representation, TensorShape, TensorTree


class Model(nn.Module, ABC):
    """One pass over a batch's inputs; the training algorithm asks for nothing else.

    Losses, activations and targets are the learner's business, not the network's: a model
    that computed them could not be exported, distilled or evaluated without one.
    """

    @abstractmethod
    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        """Named outputs, one per task the model serves, beside the features they were read from."""
        raise NotImplementedError

    def produces(self, task: str) -> Representation:
        """What this network's numbers for one task are, where that is not a plain projection.

        Shaped like ``parameters_of`` below: a default that is right for every network the framework
        composes itself, and the family that can answer better overrides it. The objective built over
        the same task is checked against this, so a head answering with angles and one answering with a
        projection cannot stand in for each other in silence.
        """
        return Representation.PROJECTED

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        """The parameters this network devotes to one task, or nothing when it shares everything.

        What a run gives a task its own learning rate over. A family that serves every task from one
        graph answers with nothing, and a rate declared for such a task is refused rather than ignored.
        """
        return ()


class Backbone(nn.Module, ABC):
    """Encodes inputs into named feature streams that heads read.

    ``feature_shapes`` is the declaration heads are sized from — shapes rather than widths, so a
    ``[C]`` vector and a ``[C, H, W]`` map are told apart before a head is built for either.
    """

    def __init__(self) -> None:
        super().__init__()
        self.carried_head: Mapping[str, Tensor] = {}
        """What a file this backbone started from carried that this backbone has no place for.

        A trained file brings a classifier, and every backbone here is built headless, so those rows
        arrive with nowhere to go. Kept rather than dropped because they are what a run growing its
        class space starts from, and read by ``build_head`` through the backbone it already holds — a
        plain mapping rather than a buffer, so that nothing a run writes down carries a head it never
        ran. Empty for a backbone built from its library's own weights, which is every ordinary run.
        """

    @property
    @abstractmethod
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """Every stream this backbone publishes, by name, without the batch axis."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        """The declared streams, in one differentiable pass."""
        raise NotImplementedError

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        """This family's own head over ``stream``, or None when it offers none.

        The library that owns the graph builds it: timm's classifier, smp's segmentation head.
        The widths are the backbone's own, so only the number of outputs is asked for. A run that
        declares ``head: native`` and receives None fails by name rather than getting another head.
        """
        return None


class Neck(nn.Module, ABC):
    """What a backbone published, brought to the shape this run's heads read.

    Between the two rather than around either, so that what a declaration named stays where it named
    it: ``backbone`` is the encoder and nothing else, ``neck`` is what a run put after it, and a path
    a ``freeze`` or an ``adapter`` writes goes on meaning what it meant before a neck was declared.
    Wrapped instead, a projection moved every path beneath it one level down, and the recipe
    ``examples/finetuning.yaml`` ships — ``modules: [backbone]`` — held that projection still along
    with the encoder, which is the one layer such a run exists to learn.

    A backbone reads a sample; a neck reads features. That is the whole of the difference, and it is
    why ``multiview`` and ``multiencoder`` are backbones — each is about how inputs are read — while
    bringing a stream to a width is not.
    """

    @property
    @abstractmethod
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """Every stream this neck publishes: the ones it brought, and the ones it passed on."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Those same streams, from the ones the backbone published."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Encoded:
    """The encoding half of a network — a backbone and whatever a run put after it — as the heads see it.

    One value rather than four arguments, because a head is built against all of it at once and the
    four travel together through every helper that sizes one: the shapes it is read from, the
    library's own classifier and its own head — both of which belong to the backbone — and which
    streams a neck published at another shape, since a library's head over one of those reads a
    feature space that is gone.

    Two fields and three derivations, because the other three are answers to what these two are. A
    fact stated twice would be free to disagree with itself the day a neck published a stream the
    backbone never did.
    """

    backbone: Backbone
    neck: Neck | None = None

    @property
    def _publishing(self) -> Backbone | Neck:
        """Whichever of the two the heads read from: the neck where a run declared one, else the backbone.

        Written once and read by both derivations below, because "which of the two" is one question and
        answering it twice is how the two answers get to disagree. Spelled ``is None`` rather than
        ``or``: an ``nn.Module`` may define ``__len__`` — measured, ``bool(nn.Sequential())`` is False —
        and a neck built from an empty one would then be passed over for the backbone in silence.
        """
        return self.backbone if self.neck is None else self.neck

    @property
    def published(self) -> Mapping[str, TensorShape]:
        """What the heads read: the neck's streams where a run declared one, the backbone's otherwise."""
        return self._publishing.feature_shapes

    @property
    def brought(self) -> frozenset[str]:
        """The streams a neck publishes at another shape than the backbone did.

        Derived rather than declared: a neck says what it publishes and nothing about what it changed,
        and the comparison is the whole of the question. A stream passed through untouched is not here,
        so the library's own head over it goes on being the head over it.
        """
        return frozenset(
            name for name, shape in self.published.items() if self.backbone.feature_shapes.get(name) != shape
        )

    @property
    def publisher(self) -> str:
        """Whose name a refusal about a stream carries: the last thing that published it."""
        return type(self._publishing).__name__


@dataclass(frozen=True, slots=True)
class HeadConnection:
    """One ready head and the features it reads, in order; the mapping key that holds it names its output.

    A tuple rather than a name, because a head reading a pair of towers is read exactly as a head
    reading one is — the model hands it what it named, in the order it named them, and how many that is
    is the declaration's business rather than the model's.
    """

    head: nn.Module
    streams: tuple[str, ...]


@runtime_checkable
class ShapeAware(Protocol):
    """A head that says which axes the feature it reads has, so a stream it cannot read is refused at build.

    Optional: any module built at ``(in_features, out_features)`` is a head, and one that declares nothing
    is simply not checked. Declaring costs one line and turns a shape error deep inside torch, a thousand
    steps into a run, into a message naming the task, the head and the stream.
    """

    reads_axes: ClassVar[tuple[str, ...]]


@runtime_checkable
class Produces(Protocol):
    """A head whose numbers are already a reading rather than a projection, and says which.

    Optional, as ``ShapeAware`` is: a head that maps the feature and does nothing else declares nothing
    and is read as a projection. Declaring costs one line and is what lets an objective needing angles
    refuse a head that cannot make them — at build, by name, rather than by how large the values
    happened to come out at the step somebody looked.
    """

    produces: ClassVar[Representation]


def produced_by(head: nn.Module) -> Representation:
    """What a head's numbers are: what the head says, or a projection where it says nothing.

    One home, because three readers ask it — the model serving a task, and each of the two wrappers
    this framework builds around a declared head, which answer for the declaration they were built
    from. A wrapper answering for itself would be saying `projected` about a tensor of angles, and the
    composition root reading that told a run to declare the head it had already declared.
    """
    return head.produces if isinstance(head, Produces) else Representation.PROJECTED


def required_input(inputs: Mapping[str, TensorTree], name: str, reader: str) -> TensorTree:
    """The input a backbone reads, refused by name with what this run actually carries.

    Three declarations have to agree on one word — `data.inputs.<name>`, `preprocessing.inputs.<name>`
    and the backbone's `input_name` — and a bare `KeyError` inside a forward pass names none of them.
    Shaped like ``Task._own``, which answers the same question on the other side of a step.
    """
    try:
        return inputs[name]
    except KeyError:
        carried = ", ".join(sorted(inputs)) or "nothing"
        raise LookupError(
            f"{reader} reads input {name!r}, and this batch carries {carried}. The name is declared in "
            "`data.inputs` and `preprocessing.inputs`; a backbone reading another says so with `input_name`."
        ) from None
