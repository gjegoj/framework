"""Port for complete models — third-party models that own their head and loss.

An *assembled* model is built from parts (``Backbone`` streams -> derived heads ->
objective criteria). A *complete* model (ultralytics YOLO, mm-style detectors, HF
checkpoints with built-in heads) arrives with head and loss fused — there is nothing
to assemble, so it bypasses the composite-model chain entirely and is driven by the
generic ``CompleteModelLitModule``. Dispatch happens in the composition root via the
``complete_models`` registry (see ``models/registry.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn

if TYPE_CHECKING:
    from src.core.entities import LossResult


class CompleteModel[PredictionT, TargetT](nn.Module, ABC):
    """A model that owns its head and loss; the framework only drives the loop.

    Generic over the decoded prediction/target pair so the model<->metric contract
    (``MetricBundle[PredictionT, TargetT]``) is checked by mypy, not by convention.

    ``family`` names the experiment assembler that knows how to build a run around
    this model (e.g. ``"detection"``); the task ``preset`` of such a run equals the
    family name.
    """

    family: ClassVar[str]

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Normalize a native batch before any forward/loss (quirk seam).

        Default is identity. Override for framework conventions the native trainer
        would otherwise apply (e.g. YOLO's uint8 -> float/255 image scaling).

        Parameters:
            batch (dict[str, Any]): Native-format batch from the run's dataloader.

        Returns:
            dict[str, Any]: The batch, normalized in place or replaced.
        """
        return batch

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Any:
        """Run the native forward pass on a prepared batch (raw native output)."""

    @abstractmethod
    def training_loss(self, batch: dict[str, Any]) -> LossResult:
        """Compute the native training loss with named components.

        Parameters:
            batch (dict[str, Any]): Prepared native batch.

        Returns:
            LossResult: Scalar total for backprop plus detached named components
            (un-namespaced — the Lightning module prefixes the task name).
        """

    @abstractmethod
    def evaluation_loss(self, batch: dict[str, Any], output: Any) -> LossResult:
        """Compute the loss from a precomputed forward ``output`` (one eval forward)."""

    @abstractmethod
    def predictions(self, output: Any) -> PredictionT:
        """Decode raw output into the metric-ready prediction form."""

    @abstractmethod
    def targets(self, batch: dict[str, Any]) -> TargetT:
        """Extract the metric-ready ground truth from the batch."""
