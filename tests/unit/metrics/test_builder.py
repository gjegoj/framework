"""``build_metric_sets``: per-stage metric sets from an explicit selection."""

from __future__ import annotations

import pytest
import torch

from src.assembly.metrics import build_metric_sets
from src.config import MetricConfig
from src.core import Objective, Stage, TargetFacts
from src.metrics import WrappedMetricSet


def test_builds_a_fresh_set_per_stage() -> None:
    sets = build_metric_sets(
        Objective.MULTICLASS, facts=TargetFacts(num_classes=3), metrics={"f1": MetricConfig(name="f1")}
    )

    assert set(sets) == set(Stage)
    assert sets[Stage.TRAIN] is not sets[Stage.VAL]


def test_no_declaration_builds_no_metrics() -> None:
    """Defaults are the preset's word, injected at config load — not the builder's."""
    sets = build_metric_sets(Objective.MULTICLASS, facts=TargetFacts(num_classes=3))

    built = sets[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    assert set(built.collection.keys()) == set()


def test_a_metric_selection_is_built_with_objective_kwargs() -> None:
    sets = build_metric_sets(
        Objective.MULTICLASS, facts=TargetFacts(num_classes=3), metrics={"f1": MetricConfig(name="f1")}
    )

    computed = sets[Stage.VAL]
    computed.update(torch.tensor([0, 1]), torch.tensor([0, 1]))

    assert set(computed.compute()) == {"f1"}


def test_unknown_metric_names_the_registered_ones() -> None:
    with pytest.raises(LookupError, match="accuracy"):
        build_metric_sets(
            Objective.MULTICLASS, facts=TargetFacts(num_classes=3), metrics={"nope": MetricConfig(name="nope")}
        )
