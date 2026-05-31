"""Objective strategies: the label semantics axis of task composition.

An objective owns the target codec, criterion, activation, output size and
metrics. It declares which topologies it supports so invalid combinations fail
with a clear message at build time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.ports import Activation, Criterion, MetricSet, TaskCodec
from src.core.registry import Registry
from src.losses.criterion import criteria
from src.metrics.builders import build_classification_metrics
from src.tasks.activations import SoftmaxActivation
from src.tasks.codecs import MulticlassTaskCodec
from src.tasks.taxonomy import Objective, Topology


class ObjectiveStrategy(ABC):
    """Produces the loss/metric/codec bricks for a given label semantics."""

    kind: Objective
    supported_topologies: frozenset[Topology]

    def supports(self, topology: Topology) -> bool:
        """Whether this objective is valid on ``topology``."""
        return topology in self.supported_topologies

    @abstractmethod
    def out_features(self, num_classes: int) -> int:
        """Number of head output channels for ``num_classes``."""

    @abstractmethod
    def build_task_codec(self) -> TaskCodec:
        """Build the task-layer target codec."""

    @abstractmethod
    def build_criterion(self) -> Criterion:
        """Build the loss criterion."""

    @abstractmethod
    def build_activation(self) -> Activation:
        """Build the logits->predictions activation for metrics/inference."""

    @abstractmethod
    def build_metrics(self, num_classes: int) -> MetricSet:
        """Build a fresh metric set sized for ``num_classes``."""


objective_strategies: Registry[ObjectiveStrategy] = Registry("objective")


@objective_strategies.register(Objective.MULTICLASS)
class MulticlassObjective(ObjectiveStrategy):
    """Mutually-exclusive classes: softmax + cross-entropy + accuracy."""

    kind = Objective.MULTICLASS
    supported_topologies = frozenset({Topology.GLOBAL, Topology.DENSE})

    def out_features(self, num_classes: int) -> int:
        return num_classes

    def build_task_codec(self) -> TaskCodec:
        return MulticlassTaskCodec()

    def build_criterion(self) -> Criterion:
        return criteria.create("cross_entropy")

    def build_activation(self) -> Activation:
        return SoftmaxActivation()

    def build_metrics(self, num_classes: int) -> MetricSet:
        return build_classification_metrics(num_classes, task="multiclass")
