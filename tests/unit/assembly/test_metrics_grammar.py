"""Metric declarations: the key is the log label, the value says which metric — always."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torchmetrics import MeanSquaredError

from src.assembly.metrics import build_metric_sets
from src.config import MetricConfig, load_config
from src.core import Objective, Stage, TargetFacts
from src.metrics import WrappedMetricSet

FACTS = TargetFacts(num_classes=3)


def train_set(metrics: dict[str, dict[str, object]] | None) -> WrappedMetricSet:
    typed = {label: MetricConfig.model_validate(params) for label, params in metrics.items()} if metrics else None
    built = build_metric_sets(Objective.MULTICLASS, facts=FACTS, metrics=typed)[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    return built


def experiment_metrics(
    metrics: dict[str, object] | None, preset: str = "classification", **extras: object
) -> dict[str, MetricConfig] | None:
    task_declaration: dict[str, object] = {"preset": preset, "target": "label", **extras}
    if metrics is not None:
        task_declaration["metrics"] = metrics
    task = load_config(
        {
            "data": {
                "source": "a.csv",
                "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                "inputs": {"image": {"column": "image"}},
            },
            "tasks": {"label": task_declaration},
            "model": {"name": "timm", "model_name": "resnet18"},
        }
    ).tasks["label"]
    return task.metrics


def segmentation_metrics(metrics: dict[str, object] | None) -> dict[str, MetricConfig] | None:
    return experiment_metrics(metrics, preset="segmentation", target_encoder={"name": "mask", "num_classes": 3})


def test_two_flavours_of_one_metric_live_under_their_own_labels() -> None:
    """The label names the log line; the metric behind it is its own declaration."""
    sets = train_set({"f1_macro": {"name": "f1", "average": "macro"}, "f1_micro": {"name": "f1", "average": "micro"}})

    computed = _updated(sets).compute()

    assert set(computed) == {"f1_macro", "f1_micro"}


def test_a_metric_that_ignores_class_facts_builds_beside_one_that_needs_them() -> None:
    """`mae` names no task or class count, so none is forced on it — offered, not forced."""
    sets = train_set({"accuracy": {"name": "accuracy"}, "mae": {"name": "mae"}})

    built = sets.collection
    assert set(built.keys()) == {"accuracy", "mae"}
    assert built["accuracy"].num_classes == FACTS.num_classes


def test_an_entry_must_say_which_metric_it_is() -> None:
    """One rule, no implicit mode: the key is a label, never a metric name."""
    with pytest.raises(ValidationError, match="exactly one"):
        experiment_metrics({"accuracy": {}})


def test_a_contradictory_entry_is_refused_at_config_load() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        experiment_metrics({"acc": {"name": "accuracy", "_target_": "torchmetrics.Accuracy"}})


def test_an_import_path_reaches_any_metric() -> None:
    """The registry is a convenience, not a gate — the docstring's promise, kept."""
    sets = train_set({"rmse": {"_target_": "torchmetrics.MeanSquaredError", "squared": False}})

    built = sets.collection
    assert isinstance(built["rmse"], MeanSquaredError)
    assert built["rmse"].squared is False


def test_derived_facts_reach_an_imported_metric_that_names_them() -> None:
    """A custom metric naming ``num_classes`` is sized like a registered one."""
    sets = train_set({"sized": {"_target_": "tests.support.fakes.SizedMetric"}})

    assert sets.collection["sized"].num_classes == FACTS.num_classes


def test_a_segmentation_preset_judges_by_overlap_out_of_the_box() -> None:
    """The kind's word arrives at load time, visible in the config, as the one grammar."""
    declared = segmentation_metrics(metrics=None)

    assert declared is not None
    assert set(declared) == {"iou", "f1", "precision", "recall", "confusion_matrix"}
    assert all(isinstance(entry, MetricConfig) for entry in declared.values())


def test_declared_metrics_silence_the_presets_word() -> None:
    """Levels replace: a set can always be narrowed, which merging could not express."""
    declared = segmentation_metrics(metrics={"iou": {"name": "iou"}})

    assert declared is not None
    assert set(declared) == {"iou"}


def test_a_kind_without_a_word_yields_no_metrics() -> None:
    """A metric-learning task is judged by its loss; nothing is injected, nothing is built."""
    declared = experiment_metrics(None, preset="metric_learning")

    assert declared is None
    built = build_metric_sets(Objective.METRIC, metrics=declared)[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    assert set(built.collection.keys()) == set()


def test_a_loaded_entry_arrives_typed_and_builds() -> None:
    declared = experiment_metrics({"top2": {"name": "accuracy", "top_k": 2}})
    assert declared is not None

    built = build_metric_sets(Objective.MULTICLASS, facts=FACTS, metrics=declared)[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    assert set(built.collection.keys()) == {"top2"}


def _updated(sets: WrappedMetricSet) -> WrappedMetricSet:
    sets.update(torch.rand(4, 3), torch.tensor([0, 1, 2, 0]))
    return sets
