"""Backbones producing the stacked views ``Stream.EMBEDDINGS`` ``[B, N, D]``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast, override

import torch
from torch import nn
from torch.nn import functional

from src.core.entities import Features
from src.core.ports import Backbone
from src.core.taxonomy import Modality, Stream
from src.models.registry import backbone_registry

if TYPE_CHECKING:
    from torch import Tensor


@backbone_registry.register("multi")
class MultiEncoderBackbone(Backbone):
    """Separate encoder per input, projected to one space and stacked ``[B, N, D]``.

    Each sub-encoder reads its own batch input and must expose ``Stream.FEATURES``; a
    per-encoder linear projection maps it to ``embedding_dim``, and the views are stacked in
    declaration order under ``Stream.EMBEDDINGS`` (CLIP/SigLIP-style dual encoders).

    Parameters:
        encoders (Mapping[str, Backbone]): Named sub-encoders, at least two; insertion order
            is the view order.
        embedding_dim (int): Width of the shared embedding space.
        normalize (bool): L2-normalize each view (the CLIP convention).
    """

    def __init__(
        self,
        encoders: Mapping[str, Backbone],
        embedding_dim: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if len(encoders) < 2:
            raise ValueError("MultiEncoderBackbone needs at least two encoders.")
        self.encoders = nn.ModuleDict(dict(encoders))
        self.projections = nn.ModuleDict(
            {name: nn.Linear(encoder.feature_dim(Stream.FEATURES), embedding_dim) for name, encoder in encoders.items()}
        )
        self._embedding_dim = embedding_dim
        self._normalize = normalize

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        views: list[Tensor] = []
        for name, encoder in self.encoders.items():
            projected = self.projections[name](encoder(inputs)[Stream.FEATURES])
            views.append(functional.normalize(projected, dim=-1) if self._normalize else projected)
        return Features(streams={Stream.EMBEDDINGS: torch.stack(views, dim=1)})

    @property
    @override
    def architecture(self) -> str:
        """What each encoder calls itself, joined — the case a config interpolation cannot reach."""
        return "+".join(cast("Backbone", encoder).architecture for encoder in self.encoders.values())

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.EMBEDDINGS: self._embedding_dim}


@backbone_registry.register("multiview")
class MultiViewBackbone(Backbone):
    """N views through one shared inner encoder, stacked into ``[B, N, D]``.

    The view axis is folded into the batch, the inner encoder runs once, and the result is
    unfolded under ``Stream.EMBEDDINGS``. Views are produced on the data side
    (``MultiViewTransform``) and not normalized here — criteria own that.

    Parameters:
        inner (Backbone): The shared encoder; must expose ``Stream.FEATURES``.
        embedding_dim (int | None): Optional per-view linear projection width; ``None``
            keeps the inner width.
        input_name (str): Which batch input holds the stacked views ``[B, N, ...]``.
    """

    def __init__(
        self,
        inner: Backbone,
        embedding_dim: int | None = None,
        input_name: str = Modality.IMAGE,
    ) -> None:
        super().__init__()
        self.inner = inner
        inner_dim = inner.feature_dim(Stream.FEATURES)
        self.projection = nn.Linear(inner_dim, embedding_dim) if embedding_dim is not None else None
        self._feature_dim = embedding_dim if embedding_dim is not None else inner_dim
        self._input_name = input_name

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        views = inputs[self._input_name]
        batch_size, view_count = views.shape[0], views.shape[1]
        flat_features = self.inner({self._input_name: views.flatten(0, 1)})[Stream.FEATURES]
        if self.projection is not None:
            flat_features = cast("Tensor", self.projection(flat_features))
        return Features(streams={Stream.EMBEDDINGS: flat_features.unflatten(0, (batch_size, view_count))})

    @property
    @override
    def architecture(self) -> str:
        """The shared encoder's: running it over N views is a way of reading, not another model."""
        return self.inner.architecture

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.EMBEDDINGS: self._feature_dim}
