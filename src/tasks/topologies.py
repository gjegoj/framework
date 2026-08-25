"""How each output structure is served: head kind and stream per ``OutputTopology`` member."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, override

from src.core.taxonomy import InputTopology, Objective, OutputTopology, Stream
from src.models import ConvHead, IdentityHead, LinearHead
from src.tasks.registry import topology_registry

if TYPE_CHECKING:
    from src.core.ports import Head


class TaskTopology(ABC):
    """The behaviour behind one ``OutputTopology`` member: head construction, stream
    choice, and which ``(objective, input)`` pairs it serves.

    The enum answers *what shape* a task's output has; a ``TaskTopology`` answers
    *how* it is produced. The stream is a method of the input axis because it is a
    joint fact: one prediction vector is read off ``FEATURES`` when one encoder
    made it and off ``EMBEDDINGS`` when several views did.
    """

    default_target_encoder: ClassVar[str | None] = None
    """The encoder this target *shape* starts from; ``None`` defers to the semantics.

    A cell's shape outranks its meaning: a dense cell is a mask file and an instances
    cell is a list of objects whatever the labels say about them, while a global cell is
    scalar-ish and only there does the objective pick the variant. The two voices meet in
    ``default_target_encoder`` in the builder, which is what assembly asks.
    """

    composes_head: ClassVar[bool] = True
    """Whether the framework builds this topology's head at all.

    Beside ``supports`` rather than inside ``build_head``, because the two are one
    question asked of a declaration — *can this framework serve this task?* — and the
    builder asks them together, before it has built anything. Stated as a refusal thrown
    from ``build_head`` instead, the answer lived inside a method the builder was never
    meant to reach, which is a promise the base class makes and one subclass breaks.
    """

    def stream(self, input_topology: InputTopology) -> str:
        """Which backbone stream carries this output's substrate."""
        return Stream.FEATURES

    @abstractmethod
    def build_head(self, in_features: int, out_features: int | None) -> Head:
        """A fresh head sized for one task; ``out_features`` is ``None`` when there is
        nothing to project and the stream itself is the output."""

    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        """Whether this output structure serves ``objective`` fed by ``input_topology``."""
        return input_topology is InputTopology.SINGLE


@topology_registry.register_instance(OutputTopology.GLOBAL)
class GlobalTopology(TaskTopology):
    """One prediction vector per sample — the one output every input arrangement feeds.

    A single encoder offers it as ``FEATURES``; several views or streams stack
    theirs into ``EMBEDDINGS``, where the head is identity and only metric
    learning supervises. A single input serves every objective, metric learning
    included: an ArcFace-style proxy judges one embedding per sample against
    class labels, which is exactly this output structure with nothing to project.
    """

    @override
    def stream(self, input_topology: InputTopology) -> str:
        return Stream.FEATURES if input_topology is InputTopology.SINGLE else Stream.EMBEDDINGS

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        # No width to project onto is the metric-learning contract: the embedding IS the output.
        return IdentityHead() if out_features is None else LinearHead(in_features, out_features)

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        # Stacked views have no per-sample labels to project onto — comparison is
        # the only supervision their carrier admits.
        return input_topology is InputTopology.SINGLE or objective is Objective.METRIC


@topology_registry.register_instance(OutputTopology.DENSE)
class DenseTopology(TaskTopology):
    """One prediction per spatial location, projected from the decoder stream.

    Metric learning never pairs with DENSE — there are no per-pixel pair or
    triplet targets — and neither does a stacked input: a decoder decodes one
    image's map.
    """

    # A dense cell is a mask file whatever the labels mean. When a depth encoder exists,
    # this becomes a joint decision of both axes — see docs/backlog.md.
    default_target_encoder: ClassVar[str | None] = "mask"

    @override
    def stream(self, input_topology: InputTopology) -> str:
        return Stream.DECODER

    def build_head(self, in_features: int, out_features: int | None) -> Head:
        if out_features is None:
            raise ValueError("A dense head projects onto classes, so it needs a width; none was asked for.")
        return ConvHead(in_features, out_features)

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        return input_topology is InputTopology.SINGLE and objective is not Objective.METRIC


@topology_registry.register_instance(OutputTopology.INSTANCES)
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
    # An instances cell is a list of objects; the boxes encoder is its one honest reading,
    # and no config line is asked for where no real choice exists.
    default_target_encoder: ClassVar[str | None] = "boxes"

    @override
    def build_head(self, in_features: int, out_features: int | None) -> Head:
        """Unreachable: ``composes_head`` is ``False``, so the builder refuses first.

        Declared only because the base class does. The explanation a user needs lives at
        the check, which is where the decision is actually taken.
        """
        raise NotImplementedError("The builder refuses a non-composing topology before reaching this.")

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        return input_topology is InputTopology.SINGLE and objective is Objective.MULTICLASS
