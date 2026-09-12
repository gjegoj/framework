"""The classification reading whose value means an image rather than a number."""

from __future__ import annotations

from typing import Any, cast

import torchmetrics
from torch import Tensor

from src.core import Matrix, Semantics
from src.metrics.registry import metric_registry


@metric_registry.register("confusion_matrix")
class ConfusionMatrix(torchmetrics.Metric):
    """How often each class was taken for each other, as a value a tracker can draw.

    torchmetrics counts; what the counts *mean* is said here, because the library has no place to say
    it: its classes subclass one another for shared state rather than for shared meaning
    (``JaccardIndex`` extends ``ConfusionMatrix``), so nothing keyed on that hierarchy would hold.

    The counting metric is held rather than inherited: torchmetrics' task wrappers dispatch in
    ``__new__`` and return a class of their own, so a subclass of one is never the class that runs.
    Holding it leaves this class without state of its own, which is also what keeps a collection from
    folding it into another metric's compute group — measured on torchmetrics 1.9.0, a metric with no
    state is never grouped.
    """

    def __init__(
        self,
        semantics: Semantics,
        num_classes: int | None = None,
        num_labels: int | None = None,
        **options: Any,
    ) -> None:
        if semantics is Semantics.MULTILABEL:
            raise ValueError(
                "A multilabel confusion matrix is one small matrix per label, which draws as nothing; "
                "drop 'confusion_matrix' from this task's metrics."
            )
        super().__init__()
        # torchmetrics types this argument as a Literal of the three words ``Semantics`` *is*, so the
        # member passes at runtime and only a type checker needs telling.
        self.counts = torchmetrics.ConfusionMatrix(
            task=cast(Any, semantics), num_classes=num_classes, num_labels=num_labels, **options
        )

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        self.counts.update(predictions, targets)

    def compute(self) -> Matrix:
        return Matrix(self.counts.compute(), xaxis="Predicted", yaxis="True")

    def reset(self) -> None:
        # Measured on torchmetrics 1.9.0: `Metric.reset` clears its own state and no child metric's.
        self.counts.reset()
        super().reset()
