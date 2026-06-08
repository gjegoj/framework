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


class PlotLogger(ABC):
    """A logger backend that can render non-scalar plots (matrices, curves).

    Scalars already reach the backend via Lightning's ``self.log`` →
    ``Logger.log_metrics``; this port adds the verbs that the scalar path cannot
    express. A concrete logger (e.g. ``ClearMLLogger``) implements both
    Lightning's ``Logger`` and this port — one object, one backend Task.
    """

    @abstractmethod
    def log_matrix(
        self,
        title: str,
        matrix: "Tensor",
        iteration: int,
        labels: list[str] | None = None,
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        """Log a 2-D matrix (e.g. confusion matrix) to the backend.

        Parameters:
            title (str): Display title / metric key.
            matrix (Tensor): 2-D float tensor on any device.
            iteration (int): Current training step (epoch or global step).
            labels (list[str] | None): Optional class label strings for both axes.
            xaxis (str | None): X-axis label (e.g. ``"Predicted"``).
            yaxis (str | None): Y-axis label (e.g. ``"True"``).
        """

    @abstractmethod
    def log_curve(
        self,
        title: str,
        x: "Tensor",
        y: "Tensor",
        iteration: int,
        series: str = "curve",
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        """Log a 2-D curve (e.g. PR curve, ROC) as a scatter/line plot.

        Parameters:
            title (str): Display title / metric key.
            x (Tensor): 1-D tensor of X-axis values (e.g. recall, FPR).
            y (Tensor): 1-D tensor of Y-axis values (e.g. precision, TPR).
            iteration (int): Current training step (epoch or global step).
            series (str): Series name within the plot (e.g. class name).
            xaxis (str | None): X-axis label.
            yaxis (str | None): Y-axis label.
        """
