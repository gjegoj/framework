"""Metric declarations: the key is the log label, the value says which metric — always; the kind fills the rest."""

from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError
from torchmetrics import MeanSquaredError

from src.config import ExperimentConfig, MetricConfig, load_config
from src.core import Stage, TaskFacts
from src.metrics import WrappedMetricSet
from src.metrics.build import build_metric_sets
from src.tasks import Classification, MetricLearning
from tests.support.configs import metrics_of
from tests.support.entities import dataset_facts

FACTS = TaskFacts(num_classes=3)


def train_set(metrics: dict[str, dict[str, object]] | None) -> WrappedMetricSet:
    typed = {label: MetricConfig.model_validate(params) for label, params in metrics.items()} if metrics else None
    built = build_metric_sets(Classification(), facts=FACTS, metrics=typed)[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    return built


def experiment(metrics: dict[str, object] | None, kind: str = "classification", **extras: object) -> ExperimentConfig:
    task_declaration: dict[str, object] = {"kind": kind, "target": "label", **extras}
    if metrics is not None:
        task_declaration["metrics"] = metrics
    return load_config(
        {
            "data": {
                "source": "a.csv",
                "split": {"train": 0.6, "val": 0.2, "test": 0.2},
                "inputs": {"image": {"column": "image"}},
            },
            "tasks": {"label": task_declaration},
            "model": {"name": "timm", "model_name": "resnet18"},
        }
    )


def experiment_metrics(
    metrics: dict[str, object] | None, kind: str = "classification", **extras: object
) -> dict[str, MetricConfig] | None:
    return experiment(metrics, kind, **extras).tasks["label"].metrics


def built_for(metrics: dict[str, object] | None, kind: str = "classification", **extras: object) -> WrappedMetricSet:
    """What a run judges the task by: the declaration, or the kind's default, built."""
    built = metrics_of(experiment(metrics, kind, **extras), dataset_facts(label=3))["label"][Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    return built


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


def test_an_imported_metric_naming_a_kinds_fact_is_handed_it() -> None:
    """A custom metric naming ``num_classes`` is sized like a registered one — by signature, the torchmetrics exception."""
    sets = train_set({"sized": {"_target_": "tests.support.fakes.SizedMetric"}})

    assert sets.collection["sized"].num_classes == FACTS.num_classes


def test_a_kinds_fact_written_on_a_metric_is_refused_by_name() -> None:
    """``num_classes`` comes from the data; a copy in config could only disagree with it."""
    with pytest.raises(ValueError, match="num_classes"):
        train_set({"accuracy": {"name": "accuracy", "num_classes": 7}})


def test_a_segmentation_kind_judges_by_overlap_out_of_the_box() -> None:
    """The kind's word is filled in at build time, through the one metric grammar."""
    built = built_for(None, kind="segmentation", target_encoder={"name": "mask", "num_classes": 3})

    assert set(built.collection.keys()) == {"iou", "f1", "precision", "recall", "confusion_matrix"}


def test_the_kinds_word_is_not_written_into_the_config() -> None:
    """Config reads only ``core``; what the kind would say stays ``None`` until the build asks it."""
    assert experiment_metrics(None, kind="segmentation", target_encoder={"name": "mask", "num_classes": 3}) is None


def test_declared_metrics_silence_the_kinds_word() -> None:
    """Levels replace: a set can always be narrowed, which merging could not express."""
    built = built_for({"iou": {"name": "iou"}}, kind="segmentation", target_encoder={"name": "mask", "num_classes": 3})

    assert set(built.collection.keys()) == {"iou"}


def test_a_kind_without_a_word_yields_no_metrics() -> None:
    """A metric-learning task is judged by its loss; nothing is filled in, nothing is built."""
    assert experiment_metrics(None, kind="metric_learning") is None
    built = built_for(None, kind="metric_learning")
    assert set(built.collection.keys()) == set()
    bare = build_metric_sets(MetricLearning())[Stage.TRAIN]
    assert isinstance(bare, WrappedMetricSet)
    assert set(bare.collection.keys()) == set()


def test_a_loaded_entry_arrives_typed_and_builds() -> None:
    declared = experiment_metrics({"top2": {"name": "accuracy", "top_k": 2}})
    assert declared is not None

    built = build_metric_sets(Classification(), facts=FACTS, metrics=declared)[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    assert set(built.collection.keys()) == {"top2"}


def _updated(sets: WrappedMetricSet) -> WrappedMetricSet:
    sets.update(torch.rand(4, 3), torch.tensor([0, 1, 2, 0]))
    return sets


def test_builds_a_fresh_set_per_stage() -> None:
    sets = build_metric_sets(Classification(), facts=TaskFacts(num_classes=3), metrics={"f1": MetricConfig(name="f1")})

    assert set(sets) == set(Stage)
    assert sets[Stage.TRAIN] is not sets[Stage.VAL]


def test_no_declaration_builds_no_metrics() -> None:
    """Defaults are the kind's, filled in by ``build_metrics``; this builder takes what it is given."""
    sets = build_metric_sets(Classification(), facts=TaskFacts(num_classes=3))

    built = sets[Stage.TRAIN]
    assert isinstance(built, WrappedMetricSet)
    assert set(built.collection.keys()) == set()


def test_a_metric_selection_is_built_with_the_kinds_kwargs() -> None:
    sets = build_metric_sets(Classification(), facts=TaskFacts(num_classes=3), metrics={"f1": MetricConfig(name="f1")})

    computed = sets[Stage.VAL]
    computed.update(torch.tensor([0, 1]), torch.tensor([0, 1]))

    assert set(computed.compute()) == {"f1"}


def test_unknown_metric_names_the_registered_ones() -> None:
    with pytest.raises(LookupError, match="accuracy"):
        build_metric_sets(Classification(), facts=TaskFacts(num_classes=3), metrics={"nope": MetricConfig(name="nope")})
