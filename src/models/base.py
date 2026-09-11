"""What a network is to this framework: named inputs in, named outputs out, and the parts a run composes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from torch import Tensor, nn

from src.core import ModelOutput, TensorShape, TensorTree


class Model(nn.Module, ABC):
    """One pass over a batch's inputs; the training algorithm asks for nothing else.

    Losses, activations and targets are the learner's business, not the network's: a model
    that computed them could not be exported, distilled or evaluated without one.
    """

    @abstractmethod
    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        """Named outputs, one per task the model serves, beside the features they were read from."""
        raise NotImplementedError

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


@dataclass(frozen=True, slots=True)
class HeadConnection:
    """One ready head and the stream it reads; the mapping key that holds it names its output."""

    head: nn.Module
    stream: str


@runtime_checkable
class ShapeAware(Protocol):
    """A head that says which axes the feature it reads has, so a stream it cannot read is refused at build.

    Optional: any module built at ``(in_features, out_features)`` is a head, and one that declares nothing
    is simply not checked. Declaring costs one line and turns a shape error deep inside torch, a thousand
    steps into a run, into a message naming the task, the head and the stream.
    """

    reads: ClassVar[tuple[str, ...]]


def reads(inputs: Mapping[str, TensorTree], name: str, reader: str) -> TensorTree:
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
