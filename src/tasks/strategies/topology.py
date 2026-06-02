"""Topology strategies: how the prediction is shaped (head + feature stream).

One of the two axes of task composition. A topology owns the head architecture
and which backbone feature stream it consumes; it knows nothing about label
semantics (that is the objective's job).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.entities import HeadSpec
from src.core.keys import DECODER, POOLED
from src.core.registry import Registry
from src.tasks.taxonomy import Topology


class TopologyStrategy(ABC):
    """Produces the head spec for a given output topology."""

    kind: Topology

    @abstractmethod
    def head_spec(self, out_features: int) -> HeadSpec:
        """Return the head build spec sized to ``out_features``."""


topology_strategies: Registry[TopologyStrategy] = Registry("topology")


@topology_strategies.register(Topology.GLOBAL)
class GlobalTopology(TopologyStrategy):
    """One prediction per sample: a linear head on the pooled feature vector."""

    kind = Topology.GLOBAL

    def head_spec(self, out_features: int) -> HeadSpec:
        return HeadSpec(kind="linear", out_features=out_features, feature_key=POOLED)


@topology_strategies.register(Topology.DENSE)
class DenseTopology(TopologyStrategy):
    """One prediction per pixel: a conv head on the decoder feature map."""

    kind = Topology.DENSE

    def head_spec(self, out_features: int) -> HeadSpec:
        return HeadSpec(kind="conv", out_features=out_features, feature_key=DECODER)
