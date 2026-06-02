"""Task head builders: small modules mapping a feature stream to logits."""

from __future__ import annotations

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
    """A 1x1 conv mapping a decoder feature map to per-pixel logits.

    Maps ``[B, in_features, H, W]`` decoder features to ``[B, out_features, H, W]``
    class logits — the dense (segmentation) counterpart of ``LinearHead``.

    Parameters:
        in_features (int): Input channel dimension (decoder stream).
        out_features (int): Output channels (class count).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        logits: Tensor = self.conv(features)
        return logits
