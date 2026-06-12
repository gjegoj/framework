"""Task head builders: small modules mapping a feature stream to logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.ports import Head
from src.models.registry import head_builders


@head_builders.register("linear")
class LinearHead(Head):
    """A single linear layer mapping a pooled feature vector to logits.

    Parameters:
        in_features (int): Input feature dimension (from the backbone).
        out_features (int): Output dimension (class count / regression dim).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor) -> Tensor:
        logits: Tensor = self.fc(features)
        return logits


@head_builders.register("conv")
class ConvHead(Head):
    """A conv layer mapping a decoder feature map to per-pixel logits.

    Maps ``[B, in_features, H, W]`` decoder features to ``[B, out_features, H, W]``
    class logits — the dense (segmentation) counterpart of ``LinearHead``.

    Defaults to ``kernel_size=3, padding=1`` (same spatial footprint as smp's
    ``SegmentationHead``); use ``kernel_size=1, padding=0`` for a pure pixel-wise
    projection.

    Parameters:
        in_features (int): Input channel dimension (decoder stream).
        out_features (int): Output channels (class count).
        kernel_size (int): Convolution kernel size (default 3).
        padding (int): Padding (default 1, keeps spatial dims unchanged).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)

    def forward(self, features: Tensor) -> Tensor:
        logits: Tensor = self.conv(features)
        return logits


@head_builders.register("cosine")
class CosineHead(Head):
    """Cosine classifier for angular-margin metric learning (ArcFace/CosFace/...).

    Produces the cosine similarity between an L2-normalized sample embedding and
    L2-normalized class prototypes — ``cos(θ)`` in ``[-1, 1]`` per class.  The
    angular margin is applied by the *loss* (e.g. ``arcface``), not here, so this
    single head serves every angular-margin method; swapping ArcFace for CosFace
    is purely a loss change.

    The prototype matrix is a learnable parameter living in the head (hence in
    ``model.heads``), so it trains and moves to device like any other head
    weight — no special wiring needed.

    Parameters:
        in_features (int): Backbone feature dimension.
        out_features (int): Number of classes (prototypes).
        embedding_dim (int | None): If set, project features to this dimension
            (bias-free) before the cosine classifier — the metric-learning
            embedding size.  ``None`` → classify directly on backbone features.
    """

    def __init__(self, in_features: int, out_features: int, embedding_dim: int | None = None) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, embedding_dim, bias=False) if embedding_dim is not None else None
        prototype_dim = embedding_dim if embedding_dim is not None else in_features
        self.weight = nn.Parameter(torch.empty(prototype_dim, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: Tensor) -> Tensor:
        embedding = self.projection(features) if self.projection is not None else features
        cosine: Tensor = F.normalize(embedding, dim=1) @ F.normalize(self.weight, dim=0)
        return cosine


@head_builders.register("identity")
class IdentityHead(Head):
    """A pass-through head that returns its input unchanged.

    Used by the MULTISTREAM topology: the per-encoder projection into the shared
    embedding space already lives in ``MultiEncoderBackbone``, so the task head
    has no work to do.  ``in_features``/``out_features`` are accepted (the head
    registry always injects them) but ignored.

    Parameters:
        in_features (int): Ignored — accepted for a uniform builder signature.
        out_features (int): Ignored — accepted for a uniform builder signature.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

    def forward(self, features: Tensor) -> Tensor:
        return features


class _AsHead(Head):
    """Adapts any ``nn.Module`` into the ``Head`` port.

    Used to wrap backbone-native heads (e.g. smp ``SegmentationHead``, timm
    classifier) and ``_target_``-instantiated custom heads so they satisfy the
    ``Head`` ABC without inheritance.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, features: Tensor) -> Tensor:
        result: Tensor = self._module(features)
        return result
