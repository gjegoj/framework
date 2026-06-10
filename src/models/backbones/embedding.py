"""Embedding backbone: an identity pass-through for precomputed feature vectors.

Streams:
    pooled [B, D] — the input embedding vector, forwarded unchanged.
"""

from __future__ import annotations

from torch import Tensor

from src.core.entities import FeatureBundle
from src.core.keys import IMAGE, POOLED
from src.core.ports import Backbone
from src.models.registry import backbones


@backbones.register("embedding")
class EmbeddingBackbone(Backbone):
    """Forwards precomputed embedding vectors as the ``pooled`` stream.

    No parameters, no encoding: the model input is already a feature vector, so the
    backbone is an identity that hands it to heads via a ``FeatureBundle``. This is
    the embedding input *modality* (plan goal G5); the task is unaware of it and sizes
    its head from ``feature_dim``.

    Parameters:
        embedding_dim (int): Dimension D of the precomputed vectors; sizes the heads.
        name (str): Ignored — accepted only so the backbone builder can pass the
            uniform ``name``/``pretrained`` arguments to every adapter.
        pretrained (bool): Ignored — there are no weights to load (see ``name``).
        input_key (str): Which input alias carries the vector (defaults to ``image``,
            so the single-input string shorthand works without extra config).
    """

    def __init__(
        self,
        embedding_dim: int,
        name: str = "identity",
        pretrained: bool = False,
        input_key: str = IMAGE,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._input_key = input_key

    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        return FeatureBundle(streams={POOLED: inputs[self._input_key]})

    def feature_dim(self, key: str) -> int:
        if key != POOLED:
            raise KeyError(f"EmbeddingBackbone exposes only {POOLED!r}, requested {key!r}.")
        return self._embedding_dim
