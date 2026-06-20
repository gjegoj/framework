"""Multi-encoder backbone: N independent sub-encoders, one named stream each.

The foundation for contrastive multi-modal training (CLIP / SIGLIP) and any
"N embedders, separate weights" setup (e.g. two photos into two distinct image
encoders).  Each sub-encoder embeds its own input and the results are exposed as
namespaced streams in the ``FeatureBundle``; a downstream ``_MultiStreamExtractor``
stacks them into ``[B, N, D]`` for a contrastive loss.

Convention: **encoder name == input alias == stream name**.  For an encoder named
``"image"`` the backbone consumes ``inputs["image"]`` and emits stream ``"image"``.

Per-encoder projection into a shared embedding space lives here (not in the head):
raw encoders differ in width, and they must agree on ``D`` before stacking.  Pass
``embed_dim`` to add a ``Linear`` after each sub-encoder; pass ``None`` when the
projection is already inside a pretrained model (CLIP/SIGLIP — sub-project B).

Streams:
    one per encoder, named after the encoder, each ``[B, embed_dim]`` (or the
    sub-encoder's pooled dim when ``embed_dim is None``).
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from src.core.entities import FeatureBundle
from src.core.keys import IMAGE, POOLED
from src.core.ports import Backbone


class MultiEncoderBackbone(Backbone):
    """Holds N sub-backbones (separate weights) plus optional per-encoder projection.

    A *composite* backbone: it does not live in the ``backbones`` registry because
    it cannot be built from a flat spec — its sub-encoders are themselves backbone
    specs needing recursive construction.  The wiring layer builds it for
    ``kind: multi`` (mirroring how context-needing callbacks are built by a
    strategy, not the flat registry).

    Parameters:
        encoders (dict[str, Backbone]): Sub-encoders keyed by name.  Each is fed
            ``{IMAGE: inputs[name]}`` so any single-input backbone works unchanged.
        embed_dim (int | None): Shared projection dimension.  ``int`` adds a
            ``Linear`` per encoder (from-scratch / mixed-domain); ``None`` passes
            each encoder's pooled output through unchanged (pretrained CLIP).
    """

    def __init__(self, encoders: dict[str, Backbone], embed_dim: int | None = None) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self._embed_dim = embed_dim
        self.projections = nn.ModuleDict(
            {name: nn.Linear(encoder.feature_dim(POOLED), embed_dim) for name, encoder in encoders.items()}
            if embed_dim is not None
            else {}
        )

    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        streams: dict[str, Tensor] = {}
        for name, encoder in self.encoders.items():
            pooled = encoder({IMAGE: inputs[name]})[POOLED]
            streams[name] = self.projections[name](pooled) if self.projections else pooled
        return FeatureBundle(streams=streams)

    def feature_dim(self, key: str) -> int:
        if key not in self.encoders:
            raise KeyError(f"MultiEncoderBackbone exposes streams {tuple(self.encoders)}, requested {key!r}.")
        if self._embed_dim is not None:
            return self._embed_dim
        encoder = cast(Backbone, self.encoders[key])
        return encoder.feature_dim(POOLED)
