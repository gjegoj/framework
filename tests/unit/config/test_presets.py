"""Task presets: familiar names as kinds of task on the config surface."""

from __future__ import annotations

import pytest

from src.config import MetricConfig, resolve_preset, task_preset_registry
from src.core import InputTopology, Objective, OutputTopology
from src.tasks.registry import objective_registry, topology_registry


@pytest.mark.parametrize(
    ("preset", "axes"),
    [
        ("classification", (OutputTopology.GLOBAL, Objective.MULTICLASS)),
        ("binary_classification", (OutputTopology.GLOBAL, Objective.BINARY)),
        ("multilabel_classification", (OutputTopology.GLOBAL, Objective.MULTILABEL)),
        ("regression", (OutputTopology.GLOBAL, Objective.CONTINUOUS)),
        ("metric_learning", (OutputTopology.GLOBAL, Objective.METRIC)),
        ("segmentation", (OutputTopology.DENSE, Objective.MULTICLASS)),
        ("binary_segmentation", (OutputTopology.DENSE, Objective.BINARY)),
        ("multilabel_segmentation", (OutputTopology.DENSE, Objective.MULTILABEL)),
        ("contrastive", (OutputTopology.GLOBAL, Objective.METRIC)),
        ("ranking", (OutputTopology.GLOBAL, Objective.METRIC)),
        ("detection", (OutputTopology.INSTANCES, Objective.MULTICLASS)),
    ],
)
def test_a_preset_is_the_familiar_name_of_one_point_on_the_axes(
    preset: str, axes: tuple[OutputTopology, Objective]
) -> None:
    """The table is the contract: two presets may share a point, and none may drift off one."""
    resolved = resolve_preset(preset)
    assert (resolved.output_topology, resolved.objective) == axes


SEGMENTATION_JUDGMENT = {"iou", "f1", "precision", "recall", "confusion_matrix"}


@pytest.mark.parametrize("preset", ["segmentation", "binary_segmentation", "multilabel_segmentation"])
def test_a_segmentation_kind_carries_its_own_judgment(preset: str) -> None:
    """Per-pixel tasks are judged by overlap beside the confusion-matrix set."""
    metrics = resolve_preset(preset).metrics

    assert metrics is not None
    assert set(metrics) == SEGMENTATION_JUDGMENT


def test_kinds_not_judged_by_per_sample_metrics_carry_none() -> None:
    """Metric learning is judged by its loss and its retrieval evals, not per-sample metrics."""
    for preset in ("metric_learning", "contrastive", "ranking"):
        assert resolve_preset(preset).metrics is None


def test_every_kinds_judgment_reads_off_the_table() -> None:
    """The design table is the contract; one home for every default."""
    assert set(resolve_preset("classification").metrics or {}) == {
        "f1",
        "precision",
        "recall",
        "confusion_matrix",
    }
    assert set(resolve_preset("regression").metrics or {}) == {"mae"}


def test_the_table_is_validated_into_the_metric_grammar_at_import() -> None:
    """A preset's word obeys the same explicit grammar as a hand-written declaration."""
    for name in task_preset_registry:
        metrics = resolve_preset(str(name)).metrics or {}
        for entry in metrics.values():
            assert isinstance(entry, MetricConfig)
            assert entry.name is not None


def test_every_registered_preset_is_pinned_by_the_table() -> None:
    pinned = {
        "classification",
        "binary_classification",
        "multilabel_classification",
        "regression",
        "metric_learning",
        "segmentation",
        "binary_segmentation",
        "multilabel_segmentation",
        "contrastive",
        "ranking",
        "detection",
    }
    assert set(task_preset_registry) == pinned


def test_unknown_preset_lists_the_known_ones() -> None:
    with pytest.raises(LookupError, match="classification"):
        resolve_preset("object_finding")


def test_presets_reference_only_implemented_axes() -> None:
    """A preset must be buildable: both of its axes carry registered behaviour."""
    for name in task_preset_registry:
        resolved = resolve_preset(str(name))
        assert resolved.output_topology in topology_registry
        assert resolved.objective in objective_registry


def test_every_preset_pairs_axes_its_topology_supports() -> None:
    """A preset naming an unsupported pairing would fail every task declared with it."""
    for name in task_preset_registry:
        resolved = resolve_preset(str(name))
        supported = topology_registry.create(resolved.output_topology).supports(
            resolved.objective, resolved.input_topology
        )
        assert supported, name


def test_only_the_paired_kinds_declare_a_non_single_input() -> None:
    """The input axis defaults to SINGLE; contrastive and ranking are the two exceptions."""
    paired = {"contrastive": InputTopology.MULTISTREAM, "ranking": InputTopology.MULTIVIEW}
    for name in task_preset_registry:
        preset = resolve_preset(str(name))
        assert preset.input_topology is paired.get(str(name), InputTopology.SINGLE), name


def test_detection_is_a_set_of_objects_judged_as_one_of_n_classes() -> None:
    """The output topology carries the geometry of a prediction, objective the semantics of its labels.

    Semantic segmentation is DENSE x MULTICLASS and depth is DENSE x CONTINUOUS, so a
    detected object's box belongs to the topology while its class is one of N like any
    other. Read the other way round, detection would become a fifth axis beside the
    three the task model has.
    """
    resolved = resolve_preset("detection")

    assert resolved.output_topology is OutputTopology.INSTANCES
    assert resolved.objective is Objective.MULTICLASS
    assert resolved.metrics is not None
    assert resolved.metrics["map"] == MetricConfig(name="map")
