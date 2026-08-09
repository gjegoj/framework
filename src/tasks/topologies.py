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

    @abstractmethod
    def build_head(self, in_features: int, out_features: int) -> Head:
        """A fresh head sized for one task."""

    def supports(self, objective: Objective) -> bool:
        """Whether this output structure can be supervised by ``objective``."""
        return True


class GlobalTopology(TaskTopology):
    """One prediction vector per sample, projected from the default stream.

    Every objective is served, metric learning included: an ArcFace-style proxy
    judges one embedding per sample against class labels, which is exactly this
    output structure with nothing to project.
    """

    def build_head(self, in_features: int, out_features: int) -> Head:
        # Zero outputs is the metric-learning contract: the embedding IS the output.
        return LinearHead(in_features, out_features) if out_features > 0 else IdentityHead()


class DenseTopology(TaskTopology):
    """One prediction per spatial location, projected from the decoder stream.

    Metric learning never pairs with DENSE — there are no per-pixel pair or
    triplet targets.
    """

    stream: ClassVar[str] = Stream.DECODER
    spatial_targets: ClassVar[bool] = True

    def build_head(self, in_features: int, out_features: int) -> Head:
        return ConvHead(in_features, out_features)

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is not Objective.METRIC


class InstancesTopology(TaskTopology):
    """A variable-length set of objects per sample — produced by the family that owns them.

    Registered although it builds nothing. A per-instance head is a vendor's: its assigner,
    its anchors and its loss are one design, and the framework composes none of them. What
    this class is for is the refusal below, which turns a `preset: detection` pointed at a
    composed backbone into a sentence naming the mismatch, instead of a shape error inside
    a head that should never have been built.
    """

    @override
    def build_head(self, in_features: int, out_features: int) -> Head:
        raise TypeError(
            "A per-instance task's head belongs to the model family that owns it — its "
            "assigner and its loss are part of the same design. Declare a vendor family "
            "instead, e.g. model: {name: yolo, architecture: yolov8n.yaml}."
        )

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is Objective.MULTICLASS


class MultiStreamTopology(TaskTopology):
    """Separate encoder per input, aligned through one stacked carrier ``[B, N, D]``.

    The multi-encoder backbone projects and stacks the views itself, so the
    head is identity; only metric learning supervises stacked views.
    """

    stream: ClassVar[str] = Stream.EMBEDDINGS

    def build_head(self, in_features: int, out_features: int) -> Head:
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

    def build_head(self, in_features: int, out_features: int) -> Head:
        return IdentityHead()

    @override
    def supports(self, objective: Objective) -> bool:
        return objective is Objective.METRIC


topology_registry.register_instance(Topology.GLOBAL, GlobalTopology())
topology_registry.register_instance(Topology.DENSE, DenseTopology())
topology_registry.register_instance(Topology.MULTISTREAM, MultiStreamTopology())
topology_registry.register_instance(Topology.MULTIVIEW, MultiViewTopology())
topology_registry.register_instance(Topology.INSTANCES, InstancesTopology())
