"""Abstract ports (interfaces) for the model, loss, metric and aggregation layers.

Inner layers depend on these ABCs; concrete adapters (timm/smp/HF backbones,
torchmetrics, ...) live in outer layers and implement them.

Parametric components that live in the autograd graph (``Backbone``, ``Head``,
``Criterion``) and stateful metric containers (``MetricSet``) inherit
``nn.Module`` — torch is the framework's "language", so this is honest rather
than leaky. Pure-logic ports (``Activation``, ``LossAggregator``) stay plain
ABCs.

Each ``nn.Module`` port re-declares a typed ``__call__`` that delegates to
``nn.Module.__call__`` (so hooks still run): torch types ``Module.__call__`` as
``Callable[..., Any]``, which erases ``forward``'s return type at every call site
(``backbone(inputs)`` would be ``Any``). The typed ``__call__`` restores it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

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

    def __call__(self, inputs: dict[str, Tensor]) -> FeatureBundle:
        # Typed delegate to nn.Module.__call__ (see module docstring); preserves hooks.
        return cast("FeatureBundle", super().__call__(inputs))

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

    def __call__(self, features: Tensor) -> Tensor:
        # Typed delegate to nn.Module.__call__ (see module docstring); preserves hooks.
        return cast("Tensor", super().__call__(features))


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

    def __call__(self, logits: Tensor, target: Tensor) -> LossResult:
        # Typed delegate to nn.Module.__call__ (see module docstring); preserves hooks.
        return cast("LossResult", super().__call__(logits, target))


class Activation(ABC):
    """Maps logits to probabilities/labels for metrics and inference (not loss)."""

    @abstractmethod
    def __call__(self, logits: Tensor) -> Tensor:
        """Convert logits to predictions used by metrics/inference."""


class TargetAdapter(ABC):
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

    @abstractmethod
    def directions(self) -> dict[str, bool | None]:
        """Return each metric's optimization direction, keyed by metric name.

        The value is the metric's intrinsic ``higher_is_better`` flag:
        ``True`` when a larger value is better (accuracy, IoU), ``False`` when
        smaller is better (error, MSE), and ``None`` when the metric has no
        direction (confusion matrix, curves). Keys match those returned by
        ``compute``, letting callers (e.g. a progress bar) bind direction to a
        metric without re-deriving it from the metric's name.
        """


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
    """A logger backend for artifacts the time-series scalar path cannot express.

    Per-step scalars already reach the backend via Lightning's ``self.log`` →
    ``Logger.log_metrics``; this port adds the verbs that path cannot express —
    matrices, curves, HTML, and end-of-run summary single-values. A concrete
    logger (e.g. ``ClearMLLogger``) implements both Lightning's ``Logger`` and
    this port — one object, one backend Task.
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

    @abstractmethod
    def log_html(self, title: str, html: str, iteration: int) -> None:
        """Log a self-contained HTML document (e.g. an interactive Plotly grid).

        Parameters:
            title (str): Display title / metric key.
            html (str): Full HTML string to ship to the backend.
            iteration (int): Current training step (epoch or global step).
        """

    @abstractmethod
    def log_single_value(self, name: str, value: float) -> None:
        """Report one final constant scalar to a summary table (no iteration axis).

        Distinct from per-step scalars: backends like ClearML collect these in a
        dedicated "Single Values" summary, suited to end-of-run headline metrics.

        Parameters:
            name (str): Display name / key for the value.
            value (float): The scalar to record.
        """

    @abstractmethod
    def log_histogram(
        self,
        title: str,
        series: str,
        values: Sequence[float],
        labels: list[str] | None = None,
    ) -> None:
        """Log a bar histogram. Calls sharing a ``title`` group as one plot.

        Used for dataset distributions: one ``series`` per stage (train/val/test)
        groups them as side-by-side bars, making class imbalance and train/test
        skew visible at a glance.

        Parameters:
            title (str): Plot title; shared across series to group them.
            series (str): Series name within the plot (e.g. the stage).
            values (Sequence[float]): Bar heights (per class or per bin).
            labels (list[str] | None): Category label per bar (x-axis ticks).
        """


@runtime_checkable
class MetricDirectionProvider(Protocol):
    """A training module that can report its metrics' optimization directions.

    A structural (not inherited) capability: any module exposing
    ``metric_directions`` satisfies it, so consumers (e.g. a progress bar) can
    colour directional deltas without reaching into the module's task graph, and
    a module lacking it degrades gracefully. Keys match those the module logs
    (``task/metric/stage``); values are each metric's ``higher_is_better`` flag.
    """

    def metric_directions(self) -> dict[str, bool | None]:
        """Return each logged metric's ``higher_is_better`` flag, by metric key."""
        ...
