"""How each output structure is served: head kind and streams per ``OutputTopology`` member."""

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
    *how* it is produced. The streams are a method of the input axis because they are a
    joint fact: one prediction vector is read off ``FEATURES`` when one encoder
    made it and off ``EMBEDDINGS`` when several views did.
    """

    default_target_encoder: ClassVar[str | None] = None
    """The encoder this target *shape* starts from; ``None`` defers to the semantics.

    A cell's shape outranks its meaning: a dense cell is a mask file and an instances
    cell is a list of objects whatever the labels say about them, while a global cell is
    scalar-ish and only there does the objective pick the variant. The two defaults meet in
    ``default_target_encoder`` in the builder, which is what assembly asks.
    """

    composes_head: ClassVar[bool] = True
    """Whether the framework has a head of its own for this output structure.

    ``False`` means the backbone's native head serves — the builder goes there without
    being asked. It is ``False`` for INSTANCES only while the only detection head is a
    backbone's; the day the framework composes one, ``build_head`` returns it and this
    flag is deleted with its readers (roadmap: *Vendor-era scaffolding*).
    """

    def streams(self, input_topology: InputTopology) -> tuple[str, ...] | None:
        """Which backbone streams carry this output's substrate, in the order a head reads
        them; ``None`` defers to the backbone's own ``pyramid()``."""
        return (Stream.FEATURES,)

    @abstractmethod
    def build_head(self, in_features: int | tuple[int, ...], out_features: int | None) -> Head:
        """A fresh head sized for one task; ``out_features`` is ``None`` when there is
        nothing to project and the stream itself is the output."""

    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        """Whether this output structure serves ``objective`` fed by ``input_topology``."""
        return input_topology is InputTopology.SINGLE


@topology_registry.register_instance(OutputTopology.GLOBAL)
class GlobalTopology(TaskTopology):
    """One prediction vector per sample — the one output every input arrangement feeds.

    A single encoder offers it as ``FEATURES``; several views or streams stack theirs into
    ``EMBEDDINGS``, where the head is identity and only metric learning supervises. A
    single input serves every objective, metric learning included (ArcFace-style proxies).
    """

    @override
    def streams(self, input_topology: InputTopology) -> tuple[str, ...] | None:
        return (Stream.FEATURES,) if input_topology is InputTopology.SINGLE else (Stream.EMBEDDINGS,)

    def build_head(self, in_features: int | tuple[int, ...], out_features: int | None) -> Head:
        width = _one_width(in_features, self)
        # No width to project onto is the metric-learning contract: the embedding IS the output.
        return IdentityHead() if out_features is None else LinearHead(width, out_features)

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        # Stacked views have no per-sample labels to project onto — comparison is
        # the only supervision stacked views admit.
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
    def streams(self, input_topology: InputTopology) -> tuple[str, ...] | None:
        return (Stream.DECODER,)

    def build_head(self, in_features: int | tuple[int, ...], out_features: int | None) -> Head:
        width = _one_width(in_features, self)
        if out_features is None:
            raise ValueError("A dense head projects onto classes, so it needs a width; none was asked for.")
        return ConvHead(width, out_features)

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        return input_topology is InputTopology.SINGLE and objective is not Objective.METRIC


@topology_registry.register_instance(OutputTopology.INSTANCES)
class InstancesTopology(TaskTopology):
    """A variable-length set of objects per sample.

    The framework has no head of its own for it: the backbone's native detection head
    serves, reading the pyramid the backbone declares and sized by the profile like every
    head; a backbone declaring no pyramid is refused by name.
    """

    composes_head: ClassVar[bool] = False
    # An instances cell is a list of objects; the boxes encoder is its one honest reading,
    # and no config line is asked for where no real choice exists.
    default_target_encoder: ClassVar[str | None] = "boxes"

    @override
    def streams(self, input_topology: InputTopology) -> tuple[str, ...] | None:
        return None  # the backbone's pyramid: its levels, its count, its order

    @override
    def build_head(self, in_features: int | tuple[int, ...], out_features: int | None) -> Head:
        """Unreachable while ``composes_head`` is ``False``: the backbone's native head serves."""
        raise NotImplementedError("The framework composes no detection head; the backbone's native head serves.")

    @override
    def supports(self, objective: Objective, input_topology: InputTopology) -> bool:
        return input_topology is InputTopology.SINGLE and objective is Objective.MULTICLASS


def _one_width(in_features: int | tuple[int, ...], topology: TaskTopology) -> int:
    """The width of the one stream a single-stream topology's head is sized from."""
    if isinstance(in_features, int):
        return in_features
    raise ValueError(
        f"{type(topology).__name__} reads one stream, but was sized from {len(in_features)} widths "
        f"{in_features}; a head over several streams belongs to a topology that reads them."
    )
