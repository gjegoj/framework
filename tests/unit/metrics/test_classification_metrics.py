"""A metric's ``compute`` returns what its value *means*, and nothing else says it.

Classes that compute a number are torchmetrics' own, registered as they come. The few
that compute an **artifact** are ours, and say so by returning one. That difference is the
whole mechanism — there is no table, no key, and nothing to keep in step with a config.

The trap it replaces is worth remembering, because these tests are what caught it:
torchmetrics subclasses for *state reuse* while changing what ``compute`` returns
(``JaccardIndex`` extends ``ConfusionMatrix``, ``AUROC`` extends the PR curve). Anything
keyed on that hierarchy hands a reduced descendant its parent's drawing and has to be
talked out of it case by case. Nothing here inherits a drawing without inheriting the
class that draws.
"""

from __future__ import annotations

from typing import Any, cast

import torch

from src.assembly.metrics import build_metric_sets
from src.config import MetricConfig
from src.core import Curve, Matrix, Objective, Stage, TargetFacts
from src.metrics import WrappedMetricSet
from src.metrics.classification import ConfusionMatrixMetric, PrecisionRecallMetric, RocMetric


def _updated(sets: WrappedMetricSet) -> WrappedMetricSet:
    sets.update(torch.rand(8, 3), torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]))
    return sets


def _multiclass(name: str, **params: Any) -> WrappedMetricSet:
    """What a config naming this metric on a three-class task actually gets.

    Extra keys become constructor arguments, exactly as they do in YAML.
    """
    sets = build_metric_sets(
        Objective.MULTICLASS,
        facts=TargetFacts(num_classes=3),
        metrics={name: MetricConfig(name=name, **params)},
    )
    return cast("WrappedMetricSet", sets[Stage.VAL])


def test_a_pr_curve_computes_as_recall_on_x_and_precision_on_y() -> None:
    computed = _updated(WrappedMetricSet({"pr": PrecisionRecallMetric(task="multiclass", num_classes=3)})).compute()

    assert isinstance(computed["pr"], Curve)
    assert (computed["pr"].xaxis, computed["pr"].yaxis) == ("Recall", "Precision")
    assert len(computed["pr"].x) == 3
    assert not computed["pr"].positive_only


def test_a_roc_computes_as_fpr_on_x_and_tpr_on_y() -> None:
    """The opposite tuple order of the PR family; read off the order, a plot would mirror."""
    computed = _updated(WrappedMetricSet({"roc": RocMetric(task="multiclass", num_classes=3)})).compute()

    assert isinstance(computed["roc"], Curve)
    assert (computed["roc"].xaxis, computed["roc"].yaxis) == ("FPR", "TPR")


def test_a_binary_curve_is_one_positive_series() -> None:
    sets = WrappedMetricSet({"roc": RocMetric(task="binary")})
    sets.update(torch.rand(8), torch.tensor([0, 1, 1, 0, 1, 0, 1, 0]))

    computed = sets.compute()["roc"]

    assert isinstance(computed, Curve)
    assert computed.positive_only
    assert len(computed.x) == 1


def test_a_confusion_matrix_computes_with_its_axes_named() -> None:
    computed = _updated(WrappedMetricSet({"cm": ConfusionMatrixMetric(task="multiclass", num_classes=3)})).compute()

    assert isinstance(computed["cm"], Matrix)
    assert (computed["cm"].xaxis, computed["cm"].yaxis) == ("Predicted", "True")
    assert computed["cm"].labels is None  # the class space; the router fills the names


def test_a_multilabel_confusion_matrix_publishes_nothing() -> None:
    """``[L, 2, 2]`` has no single plot. Identified means a quiet skip, never a warning
    once an epoch about a shape the reader has no way to change."""
    sets = WrappedMetricSet({"cm": ConfusionMatrixMetric(task="multilabel", num_labels=3)})
    sets.update(torch.rand(8, 3), torch.randint(0, 2, (8, 3)))

    assert sets.compute() == {}


def test_the_facts_an_objective_offers_reach_a_wrapped_metric() -> None:
    """A wrapper names ``task`` and the class counts, and that is load-bearing.

    Assembly offers derived values to whatever *names* them, so a constructor of
    ``**kwargs`` alone receives none of them. Written that way first, every wrapped
    metric was built with no task mode and no class count, and died inside torchmetrics
    about an argument no config had mentioned. Measured: 27 tests across assembly.
    """
    computed = _updated(_multiclass("roc")).compute()["roc"]

    assert isinstance(computed, Curve)
    assert len(computed.x) == 3  # the class count arrived, from the profile alone


def test_scalar_metrics_pass_through_untouched() -> None:
    """A metric computing a number needs nothing from us, so it is torchmetrics' own class."""
    computed = _updated(_multiclass("accuracy")).compute()["accuracy"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 0


def test_iou_computes_as_a_per_class_vector_not_a_matrix() -> None:
    """The segmentation preset's flagship metric, and the reason the old table needed guards.

    ``JaccardIndex`` *is* a ``ConfusionMatrix`` subclass in torchmetrics. It draws nothing
    here for one reason: it is registered as the plain class, and only a class of ours
    draws. There is no geometry check anywhere that could be forgotten.
    """
    computed = _updated(_multiclass("iou", average="none")).compute()["iou"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 1


def test_multilabel_iou_is_not_swallowed_by_the_confusion_matrix() -> None:
    sets = build_metric_sets(
        Objective.MULTILABEL,
        facts=TargetFacts(num_classes=3),
        metrics={"iou": MetricConfig(name="iou")},
    )[Stage.VAL]
    sets.update(torch.rand(8, 3), torch.randint(0, 2, (8, 3)))

    assert "iou" in sets.compute()


def test_a_scalar_descendant_of_a_curve_family_keeps_its_scalar() -> None:
    """``AUROC`` subclasses the PR curve for state reuse; unwrapped, it cannot draw."""
    from torchmetrics import AUROC

    sets = WrappedMetricSet({"auroc": AUROC(task="binary")})
    sets.update(torch.rand(8), torch.randint(0, 2, (8,)))

    computed = sets.compute()["auroc"]

    assert isinstance(computed, torch.Tensor)
    assert computed.ndim == 0


def test_a_subclass_inherits_the_drawing_of_the_class_it_extends() -> None:
    """A tuned metric draws like the one it extends, by ordinary inheritance rather than
    by a lookup that reimplements it."""

    class TunedPr(PrecisionRecallMetric):
        """Someone's own thresholds, same meaning."""

    assert isinstance(
        _updated(WrappedMetricSet({"pr": TunedPr(task="multiclass", num_classes=3)})).compute()["pr"], Curve
    )


def test_an_unknown_metrics_value_passes_through_untouched() -> None:
    """No drawing claimed, no guessing — the router warns about the raw artifact by name."""
    from torchmetrics import Metric

    class TupleMetric(Metric):
        def update(self, predictions: torch.Tensor, target: torch.Tensor) -> None: ...

        def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.5]))

    sets = WrappedMetricSet({"odd": TupleMetric()})
    sets.update(torch.rand(2), torch.rand(2))

    assert isinstance(sets.compute()["odd"], tuple)
