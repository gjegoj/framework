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
