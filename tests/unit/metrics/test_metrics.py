"""What a run is judged by: the names a declaration writes, and the one reading that means an image."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from src.core import Matrix, Semantics, require_tensor
from src.metrics import ConfusionMatrix
from src.metrics.build import build_metrics
from src.metrics.registry import metric_registry
from src.tasks.registry import task_registry
from tests.support.tasks import specimen

CLASSES = 3
VOTES = torch.rand(4, CLASSES).softmax(dim=1)
CHOICES = torch.tensor([0, 1, 2, 0])
NUMBERS = torch.tensor([1.5, 2.5, 3.5, 4.5])
VOCABULARY: Mapping[str, object] = {"semantics": Semantics.MULTICLASS, "num_classes": CLASSES}

SPECIMENS: dict[str, tuple[Mapping[str, object], Tensor, Tensor]] = {
    "accuracy": (VOCABULARY, VOTES, CHOICES),
    "f1": (VOCABULARY, VOTES, CHOICES),
    "precision": (VOCABULARY, VOTES, CHOICES),
    "recall": (VOCABULARY, VOTES, CHOICES),
    "confusion_matrix": (VOCABULARY, VOTES, CHOICES),
    "iou": (VOCABULARY, torch.rand(4, CLASSES, 6, 6), torch.zeros(4, 6, 6, dtype=torch.long)),
    "mae": ({}, NUMBERS, NUMBERS + 1),
    "mse": ({}, NUMBERS, NUMBERS + 1),
}
"""The facts each name is sized by and a batch it scores: a newly registered name needs a row here."""


class TestContract:
    @pytest.mark.parametrize("name", list(metric_registry))
    def test_every_registered_name_scores_the_batch_its_facts_describe(self, name: str) -> None:
        assert name in SPECIMENS, f"{name!r} is registered but has no specimen batch; add one to SPECIMENS."
        facts, predictions, targets = SPECIMENS[name]

        metrics = build_metrics({name: {"name": name}}, facts)
        metrics.update(predictions, targets)
        computed = metrics.compute()

        assert set(computed) == {name} and isinstance(computed[name], Tensor | Matrix)

    @pytest.mark.parametrize("kind", list(task_registry))
    def test_the_metrics_a_kind_declares_score_what_that_kind_predicts(self, kind: str) -> None:
        """A task's table and the views it answers with have to fit each other, whatever its semantics."""
        task, output, batch = specimen(kind)
        declared = type(task).default_metrics

        metrics = build_metrics(declared, task.facts())
        metrics.update(require_tensor(task.postprocess(output), name=task.name), task.metric_view(batch))

        assert set(metrics.compute()) == set(declared)

    def test_two_flavours_of_one_metric_stand_side_by_side(self) -> None:
        """The label is what a report shows, so it is the declaration's to choose, not the metric's."""
        metrics = build_metrics(
            {"f1": {"name": "f1", "average": "macro"}, "f1_per_class": {"name": "f1", "average": "none"}}, VOCABULARY
        )
        metrics.update(VOTES, CHOICES)

        computed = metrics.compute()

        assert computed["f1"].ndim == 0 and computed["f1_per_class"].shape == (CLASSES,)


class TestConfusionMatrix:
    def test_it_says_which_of_its_axes_is_the_prediction(self) -> None:
        """A bare square tensor is any two-dimensional reading; which axis is which is this class's to say."""
        matrix = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)
        matrix.update(VOTES, CHOICES)

        drawn = matrix.compute()

        assert isinstance(drawn, Matrix) and drawn.value.shape == (CLASSES, CLASSES)
        assert (drawn.xaxis, drawn.yaxis) == ("Predicted", "True")

    def test_it_forgets_its_counts_with_the_metric_that_holds_them(self) -> None:
        """Measured on torchmetrics 1.9.0: resetting a metric leaves the state of a metric inside it."""
        matrix = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)
        matrix.update(VOTES, CHOICES)

        matrix.reset()
        matrix.update(VOTES, CHOICES)

        assert matrix.compute().value.sum() == len(CHOICES)

    def test_it_is_never_folded_into_another_metrics_compute_group(self) -> None:
        """Why it holds its counter instead of inheriting one: a metric with no state of its own is never
        grouped (torchmetrics 1.9.0). Were that to change, a collection would update only the group's
        leader and this matrix would draw stale counts — a wrong chart, not an error."""
        grouped = build_metrics({"f1": {"name": "f1"}, "confusion_matrix": {"name": "confusion_matrix"}}, VOCABULARY)
        alone = ConfusionMatrix(semantics=Semantics.MULTICLASS, num_classes=CLASSES)

        for _ in range(2):
            grouped.update(VOTES, CHOICES)
            alone.update(VOTES, CHOICES)

        assert torch.equal(grouped.compute()["confusion_matrix"].value, alone.compute().value)

    def test_a_multilabel_matrix_is_refused_where_it_was_declared(self) -> None:
        """One small matrix per label draws as nothing; saying so before the run beats logging silence."""
        with pytest.raises(ValueError, match="multilabel"):
            build_metrics(
                {"confusion_matrix": "confusion_matrix"},
                {"semantics": Semantics.MULTILABEL, "num_classes": CLASSES},
            )
