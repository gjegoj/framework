"""A declaration becomes a task's metrics: options come from the run, sizes from what its target settled."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torchmetrics import Metric

from src.config import ComponentConfig
from src.core import Semantics
from src.metrics.build import build_metrics

CLASSES = 3
VOCABULARY: Mapping[str, object] = {"semantics": Semantics.MULTICLASS, "num_classes": CLASSES}
VOTES = torch.rand(4, CLASSES).softmax(dim=1)
CHOICES = torch.tensor([0, 1, 2, 0])


class TestFacts:
    def test_a_fact_reaches_only_the_metric_that_names_it(self) -> None:
        """`mae` stands beside `accuracy` in a multitask run and would refuse the vocabulary it is not about."""
        metrics = build_metrics({"accuracy": {"name": "accuracy"}, "mae": {"name": "mae"}}, VOCABULARY)

        assert set(metrics) == {"accuracy", "mae"}

    def test_a_fact_the_target_already_settled_is_refused_where_it_was_restated(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            build_metrics({"f1": {"name": "f1", "num_classes": CLASSES}}, VOCABULARY)

    def test_a_fact_torchmetrics_has_no_word_for_still_reaches_a_metric_that_names_it(self) -> None:
        """One table of facts, two builders: what a loss is offered, a metric is offered too.

        Translating into the library's dialect used to *replace* the run's facts with four keys of
        torchmetrics', so a metric of the framework's own — or a reader's, reached by `_target_` —
        could name nothing outside them. The bin centres a binned target settles were the first
        casualty: no metric could read the numbers its own task is about.
        """
        declared = {"spread": {"_target_": "tests.unit.metrics.test_build.Spread"}}

        built = build_metrics(declared, {**VOCABULARY, "values": (0.0, 10.0, 20.0)})

        assert built["spread"].values == (0.0, 10.0, 20.0)

    def test_a_metric_the_task_cannot_size_is_named_with_what_it_was_offered(self) -> None:
        """Overlap between predicted and true pixels means nothing for a number; the refusal says so by name."""
        with pytest.raises(ValueError, match="iou"):
            build_metrics({"iou": {"name": "iou"}}, {})


class Spread(Metric):
    """A metric of the framework's own, about the numbers a binned target stands for."""

    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.values = values

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        return None

    def compute(self) -> torch.Tensor:
        return torch.tensor(max(self.values) - min(self.values))


class TestDeclaration:
    @pytest.mark.parametrize(
        "declared",
        [
            pytest.param("mae", id="a bare name, as a task writes its own default"),
            pytest.param({"name": "mae"}, id="a mapping, as a config writes it"),
            pytest.param(ComponentConfig(name="mae"), id="a validated component, as a run hands it over"),
        ],
    )
    def test_a_declaration_is_read_in_any_of_the_forms_it_is_written_in(self, declared: object) -> None:
        assert set(build_metrics({"mae": declared}, {})) == {"mae"}

    def test_declared_options_reach_the_metric(self) -> None:
        metrics = build_metrics({"f1": {"name": "f1", "average": "none"}}, VOCABULARY)
        metrics.update(VOTES, CHOICES)

        assert metrics.compute()["f1"].shape == (CLASSES,)

    def test_a_metric_torchmetrics_offers_needs_no_registration(self) -> None:
        """The registry lists the names worth writing; anything the library has is one import path away."""
        metrics = build_metrics({"kappa": {"_target_": "torchmetrics.CohenKappa"}}, VOCABULARY)
        metrics.update(VOTES, CHOICES)

        assert metrics.compute()["kappa"].ndim == 0

    def test_a_task_judged_by_nothing_has_nothing_to_compute(self) -> None:
        assert len(build_metrics({}, VOCABULARY)) == 0
