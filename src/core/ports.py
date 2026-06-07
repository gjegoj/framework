"""Abstract ports (interfaces) for the model, loss, metric and aggregation layers.

Inner layers depend on these ABCs; concrete adapters (timm/smp/HF backbones,
torchmetrics, ...) live in outer layers and implement them.

Parametric components that live in the autograd graph (``Backbone``, ``Head``,
``Criterion``) and stateful metric containers (``MetricSet``) inherit
``nn.Module`` — torch is the framework's "language", so this is honest rather
than leaky. Pure-logic ports (``Activation``, ``LossAggregator``) stay plain
ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from src.core.entities import FeatureBundle, LossResult, TargetView


class Backbone(nn.Module, ABC):
    """Encodes raw model inputs into named feature streams."""

    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        """Encode ``inputs`` into a ``FeatureBundle`` of named streams.

        Parameters:
            inputs (dict[str, Tensor]): Batched, named model inputs.

        Returns:
            FeatureBundle: Named feature streams consumed by heads.
        """

    @abstractmethod
    def feature_dim(self, key: str) -> int:
        """Return the channel/feature dimension of stream ``key`` (sizes heads)."""

    def native_head(self, feature_key: str, in_features: int, out_features: int) -> nn.Module | None:
        """Return a backbone-native head for ``feature_key``, or ``None``.

        Override in concrete backbones to expose the architecture's own head
        (e.g. smp's ``SegmentationHead``, timm's ``create_classifier``).
        Returning ``None`` falls back to the head registry.
        """
        return None


class Head(nn.Module, ABC):
    """Maps one selected feature stream to task logits."""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Map a feature stream to raw logits (pre-activation) for the task."""


class Criterion(nn.Module, ABC):
    """Computes a task loss from logits and target (operates on logits)."""

    @abstractmethod
    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        """Compute named loss components and their total.

        Parameters:
            logits (Tensor): Raw model outputs (pre-activation).
            target (Tensor): Ground-truth target shaped for this loss.

        Returns:
            LossResult: Total scalar loss and its named components.
        """


class Activation(ABC):
    """Maps logits to probabilities/labels for metrics and inference (not loss)."""

    @abstractmethod
    def __call__(self, logits: Tensor) -> Tensor:
        """Convert logits to predictions used by metrics/inference."""


class TaskCodec(ABC):
    """Adapts a raw batched target into loss/metric views (task-layer shaping)."""

    @abstractmethod
    def adapt(self, target: Tensor) -> TargetView:
        """Shape/type the raw target for this task's loss and metrics."""


class MetricSet(nn.Module, ABC):
    """A stateful collection of metrics for one task and stage."""

    @abstractmethod
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate one batch of predictions against targets."""

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Return computed metric values keyed by metric name."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state at the start of an epoch."""


class LossAggregator(ABC):
    """Combines per-task losses into a single optimization objective."""

    @abstractmethod
    def combine(self, losses: dict[str, LossResult], weights: dict[str, float]) -> LossResult:
        """Aggregate per-task losses (e.g. weighted sum) into one ``LossResult``.

        Parameters:
            losses (dict[str, LossResult]): Per-task loss results by task name.
            weights (dict[str, float]): Per-task weights by task name.

        Returns:
            LossResult: Combined total plus per-task components for logging.
        """
