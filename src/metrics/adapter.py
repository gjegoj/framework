"""Adapter wrapping a torchmetrics ``MetricCollection`` behind the MetricSet port.

Being an ``nn.Module`` (via ``MetricSet``), the wrapped collection moves with the
Lightning module onto the accelerator and tracks state per stage.
"""

from __future__ import annotations

from typing import Any

from torchmetrics import MetricCollection

from src.core.ports import MetricSet


class TorchMetricsAdapter(MetricSet):
    """Wraps a ``MetricCollection`` so it satisfies the ``MetricSet`` port.

    Parameters:
        collection (MetricCollection): The torchmetrics collection to wrap.
    """

    def __init__(self, collection: MetricCollection) -> None:
        super().__init__()
        self.collection = collection

    def update(self, preds: Any, target: Any) -> None:
        self.collection.update(preds, target)

    def compute(self) -> dict[str, Any]:
        return dict(self.collection.compute())

    def reset(self) -> None:
        self.collection.reset()

    def directions(self) -> dict[str, bool | None]:
        return {name: metric.higher_is_better for name, metric in self.collection.items()}
