"""How each output structure is served: head kind and stream per ``Topology`` member."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, override

from src.core.taxonomy import Objective, Stream, Topology
from src.models import ConvHead, IdentityHead, LinearHead
from src.tasks.registry import topology_registry

if TYPE_CHECKING:
    from src.core.ports import Head


class TaskTopology(ABC):
    """The behaviour behind one ``Topology`` member: head construction and stream choice.

    The enum answers *what shape* a task's output has; a ``TaskTopology``
    answers *how* it is produced — which head kind and which feature stream.
    """

    stream: ClassVar[str] = Stream.FEATURES

    spatial_targets: ClassVar[bool] = False
    """Whether targets live in image space, which no objective can encode on its own.

    A per-pixel target is a file of its own, and reading it needs facts config
    has to state (the class count, a root path). Such a task therefore declares
    its encoder instead of inheriting the objective's default.
    """

    composes_head: ClassVar[bool] = True
    """Whether the framework builds this topology's head at all.

    Beside ``supports`` rather than inside ``build_head``, because the two are one
    question asked of a declaration — *can this framework serve this task?* — and the
    builder asks them together, before it has built anything. Stated as a refusal thrown
    from ``build_head`` instead, the answer lived inside a method the builder was never
    meant to reach, which is a promise the base class makes and one subclass breaks.
    """

    @abstractmethod
    def build_head(self, in_features: int, out_features: int | None) -> Head:
        """A fresh head sized for one task; ``out_features`` is ``None`` when there is
        nothing to project and the stream itself is the output."""

    def supports(self, objective: Objective) -> bool:
        """Whether this output structure can be supervised by ``objective``."""
        return True


class GlobalTopology(TaskTopology):
    """One prediction vector per sample, projected from the default stream.

    Every objective is served, metric learning included: an ArcFace-style proxy
    judges one embedding per sample against class labels, which is exactly this
    output structure with nothing to project.
    """

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        # No width to project onto is the metric-learning contract: the embedding IS the output.
        return IdentityHead() if out_features is None else LinearHead(in_features, out_features)


class DenseTopology(TaskTopology):
    """One prediction per spatial location, projected from the decoder stream.

    Metric learning never pairs with DENSE — there are no per-pixel pair or
    triplet targets.
    """

    stream: ClassVar[str] = Stream.DECODER
    spatial_targets: ClassVar[bool] = True

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        if out_features is None:
            raise ValueError("A dense head projects onto classes, so it needs a width; none was asked for.")
        return ConvHead(in_features, out_features)

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is not Objective.METRIC


class InstancesTopology(TaskTopology):
    """A variable-length set of objects per sample — produced by the family that owns them.

    Registered although it composes nothing. A per-instance head is a vendor's: its
    assigner, its anchors and its loss are one design, and the framework composes none of
    them. What this class is for is to say so — ``composes_head = False`` — so that a
    ``preset: detection`` pointed at a composed backbone is refused by the builder, in a
    sentence naming the mismatch, rather than by a shape error inside a head that should
    never have been built.
    """

    composes_head: ClassVar[bool] = False

    @override
    def build_head(self, in_features: int, out_features: int | None) -> Head:
        """Unreachable: ``composes_head`` is ``False``, so the builder refuses first.

        Declared only because the base class does. The explanation a user needs lives at
        the check, which is where the decision is actually taken.
        """
        raise NotImplementedError("The builder refuses a non-composing topology before reaching this.")

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is Objective.MULTICLASS


class MultiStreamTopology(TaskTopology):
    """Separate encoder per input, aligned through one stacked carrier ``[B, N, D]``.

    The multi-encoder backbone projects and stacks the views itself, so the
    head is identity; only metric learning supervises stacked views.
    """

    stream: ClassVar[str] = Stream.EMBEDDINGS

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        return IdentityHead()

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is Objective.METRIC


class MultiViewTopology(TaskTopology):
    """N views of each sample through one shared encoder, stacked ``[B, N, D]``.

    Behaviourally a twin of ``MultiStreamTopology`` today — views of one
    sample instead of modality streams. Kept separate because the axes are
    different concepts and may diverge (e.g. supervised in-batch mining).
    """

    stream: ClassVar[str] = Stream.EMBEDDINGS

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        return IdentityHead()

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is Objective.METRIC


topology_registry.register_instance(Topology.GLOBAL, GlobalTopology())
topology_registry.register_instance(Topology.DENSE, DenseTopology())
topology_registry.register_instance(Topology.MULTISTREAM, MultiStreamTopology())
topology_registry.register_instance(Topology.MULTIVIEW, MultiViewTopology())
topology_registry.register_instance(Topology.INSTANCES, InstancesTopology())
