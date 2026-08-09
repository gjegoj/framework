"""The presentation table turns torchmetrics values into identified artifacts.

PR and ROC tuples share one geometry with opposite axis orientation —
``(precision, recall, _)`` against ``(fpr, tpr, _)`` — so orientation cannot
be a dispatch-time guess. The table is keyed by metric class, looked up along
the MRO, and open: one registered translator per family, ours or a user's.
"""

from __future__ import annotations

import torch
from torchmetrics import ROC, ConfusionMatrix, PrecisionRecallCurve

from src.core import Curve, Matrix
from src.metrics import WrappedMetricSet


def _updated(sets: WrappedMetricSet) -> WrappedMetricSet:
    sets.update(torch.rand(8, 3), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
    return sets


def test_a_pr_curve_computes_as_recall_on_x_and_precision_on_y() -> None:
    sets = WrappedMetricSet({"pr": PrecisionRecallCurve(task="multiclass", num_classes=3)})

    computed = _updated(sets).compute()["pr"]

    assert isinstance(computed, Curve)
    assert (computed.xaxis, computed.yaxis) == ("Recall", "Precision")
    assert len(computed.x) == 3
    assert not computed.positive_only


def test_a_roc_computes_as_fpr_on_x_and_tpr_on_y() -> None:
    """The opposite tuple order of the PR family; a geometry guess would mirror the plot."""
    sets = WrappedMetricSet({"roc": ROC(task="multiclass", num_classes=3)})

    computed = _updated(sets).compute()["roc"]

    assert isinstance(computed, Curve)
    assert (computed.xaxis, computed.yaxis) == ("FPR", "TPR")


def test_a_binary_curve_is_one_positive_series() -> None:
    sets = WrappedMetricSet({"roc": ROC(task="binary")})
    sets.update(torch.rand(8), torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]))

    computed = sets.compute()["roc"]

    assert isinstance(computed, Curve)
    assert computed.positive_only
    assert len(computed.x) == 1


def test_scalar_metrics_pass_through_untouched() -> None:
    from torchmetrics import Accuracy

    sets = WrappedMetricSet({"accuracy": Accuracy(task="multiclass", num_classes=3)})

    computed = _updated(sets).compute()["accuracy"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 0


def test_the_most_derived_class_wins_the_mro_walk() -> None:
    """ROC subclasses the PR curve in torchmetrics; presentation must follow the subclass."""
    computed = _updated(WrappedMetricSet({"roc": ROC(task="multiclass", num_classes=3)})).compute()["roc"]

    assert (computed.xaxis, computed.yaxis) == ("FPR", "TPR")


def test_a_subclass_without_its_own_presentation_inherits_the_parents() -> None:
    """LSP as a mechanism: a tuned metric presents like its family until it says otherwise."""
    from torchmetrics.classification import MulticlassPrecisionRecallCurve

    class TunedPr(MulticlassPrecisionRecallCurve): ...

    sets = WrappedMetricSet({"pr": TunedPr(num_classes=3)})

    assert isinstance(_updated(sets).compute()["pr"], Curve)


def test_a_confusion_matrix_presents_with_its_axes() -> None:
    sets = WrappedMetricSet({"cm": ConfusionMatrix(task="multiclass", num_classes=3)})

    computed = _updated(sets).compute()["cm"]

    assert isinstance(computed, Matrix)
    assert (computed.xaxis, computed.yaxis) == ("Predicted", "True")
    assert computed.labels is None  # the class space; the router fills the names


def test_multilabel_confusion_is_identified_but_not_drawable() -> None:
    """[L, 2, 2] has no single plot; identified means a quiet skip, never an epoch-wise warning."""
    sets = WrappedMetricSet({"cm": ConfusionMatrix(task="multilabel", num_labels=3)})
    sets.update(torch.rand(8, 3), torch.randint(0, 2, (8, 3)))

    assert sets.compute() == {}


def test_an_unknown_metrics_value_passes_through_untouched() -> None:
    """No translator, no guessing — the router will warn about the raw artifact."""
    from torchmetrics import Metric

    class TupleMetric(Metric):
        def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None: ...

        def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.5]))

    sets = WrappedMetricSet({"odd": TupleMetric()})
    sets.update(torch.rand(2), torch.rand(2))

    assert isinstance(sets.compute()["odd"], tuple)


def test_iou_computes_as_a_per_class_vector_not_a_matrix() -> None:
    """torchmetrics implements JaccardIndex as a ConfusionMatrix subclass; the segmentation
    preset's flagship metric must not inherit a presentation for a geometry it no longer has."""
    from torchmetrics import JaccardIndex

    sets = WrappedMetricSet({"iou": JaccardIndex(task="multiclass", num_classes=3, average="none")})

    computed = _updated(sets).compute()["iou"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 1


def test_multilabel_iou_is_not_swallowed_by_the_confusion_translator() -> None:
    from torchmetrics import JaccardIndex

    sets = WrappedMetricSet({"iou": JaccardIndex(task="multilabel", num_labels=3)})
    sets.update(torch.rand(8, 3), torch.randint(0, 2, (8, 3)))

    assert "iou" in sets.compute()


def test_a_scalar_descendant_of_a_curve_family_keeps_its_scalar() -> None:
    """AUROC subclasses the PR curve for state reuse; its 0-D value must pass through."""
    from torchmetrics import AUROC

    sets = WrappedMetricSet({"auroc": AUROC(task="binary")})
    sets.update(torch.rand(8), torch.randint(0, 2, (8,)))

    computed = sets.compute()["auroc"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 0
