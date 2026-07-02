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
from src.core.taxonomy import Topology


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
        return HeadSpec(kind="linear", out_features=out_features, feature_key=POOLED, prefer_native=True)


@topology_strategies.register(Topology.DENSE)
class DenseTopology(TopologyStrategy):
    """One prediction per pixel: a conv head on the decoder feature map."""

    kind = Topology.DENSE

    def head_spec(self, out_features: int) -> HeadSpec:
        return HeadSpec(kind="conv", out_features=out_features, feature_key=DECODER, prefer_native=True)


@topology_strategies.register(Topology.RANKING)
class RankingTopology(TopologyStrategy):
    """N views of the same input through a shared backbone (Siamese network).

    The backbone is called once with a stacked ``[B*N, ...]`` batch; the
    assembly layer reshapes the output to ``[B, N, D]`` before the head.
    ``nn.Linear`` applies to the last dimension, so the same projection head
    works without modification for any N.

    ``view_keys`` names which input aliases to stack.  In config-driven
    experiments this is left ``None`` and derived from ``data.inputs`` by the
    wiring layer — the data config is the single source of truth for input names.
    Pass explicit keys only for programmatic construction (tests, scripts).

    Parameters:
        view_keys (tuple[str, ...] | None): Input alias names matching
            ``data.inputs`` keys.  ``None`` → wiring derives them at build time.

    Streams:
        POOLED ``[B, D]`` — backbone pooled output (same as GlobalTopology).
    """

    kind = Topology.RANKING

    def __init__(self, view_keys: tuple[str, ...] | None = None) -> None:
        self.view_keys = view_keys

    def head_spec(self, out_features: int) -> HeadSpec:
        return HeadSpec(
            kind="linear",
            out_features=out_features,
            feature_key=POOLED,
            prefer_native=False,
            view_keys=self.view_keys,
        )


@topology_strategies.register(Topology.MULTISTREAM)
class MultistreamTopology(TopologyStrategy):
    """N streams from N separate encoders (dual/multi-encoder, e.g. CLIP/SIGLIP).

    Unlike RANKING (one shared backbone over stacked input views), here a
    ``MultiEncoderBackbone`` produces N named streams with *separate* weights.
    The assembly layer stacks the named streams into ``[B, N, D]``; per-encoder
    projection into the shared space lives in the backbone, so the task head is
    identity.

    ``stream_keys`` names which ``FeatureBundle`` streams to stack.  In
    config-driven experiments it is left ``None`` and derived from ``data.inputs``
    by the wiring layer (encoder name == input alias == stream name).  Pass
    explicit keys only for programmatic construction (tests, scripts).

    Parameters:
        stream_keys (tuple[str, ...] | None): Ordered stream names to stack.
            ``None`` → wiring derives them at build time.

    Streams:
        the named encoder streams ``[B, D]`` each → stacked ``[B, N, D]``.
    """

    kind = Topology.MULTISTREAM

    def __init__(self, stream_keys: tuple[str, ...] | None = None) -> None:
        self.stream_keys = stream_keys

    def head_spec(self, out_features: int) -> HeadSpec:
        return HeadSpec(
            kind="identity",
            out_features=out_features,
            feature_key=POOLED,
            prefer_native=False,
            stream_keys=self.stream_keys,
        )
