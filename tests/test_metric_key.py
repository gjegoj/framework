"""Tests for the MetricKey value object — the single parser of logged-scalar keys.

The grammar (``{task}/{metric}/{stage}[/leaf]`` for task metrics,
``loss/{stage}/{component}`` for losses) is parsed in exactly one place; these
tests pin every shape the live consumers (summary / progress bar / ClearML)
depend on.
"""

from __future__ import annotations

from src.core.enums import Stage
from src.core.metric_key import MetricKey


class TestParseLoss:
    def test_aggregate_loss(self) -> None:
        key = MetricKey.parse("loss/val/total")
        assert key.is_loss is True
        assert key.stage == Stage.VAL
        assert key.task is None
        assert key.name == "total"
        assert key.leaf is None

    def test_per_task_loss_component_keeps_slashed_name(self) -> None:
        key = MetricKey.parse("loss/train/breed/cross_entropy")
        assert key.is_loss is True
        assert key.stage == Stage.TRAIN
        assert key.name == "breed/cross_entropy"

    def test_bare_loss_token_is_unclassified(self) -> None:
        # 'loss' alone is not the loss/{stage}/{component} form.
        key = MetricKey.parse("loss")
        assert key.is_loss is False
        assert key.stage is None


class TestParseMetric:
    def test_scalar_metric(self) -> None:
        key = MetricKey.parse("species/f1/val")
        assert key.is_loss is False
        assert key.task == "species"
        assert key.name == "f1"
        assert key.stage == Stage.VAL
        assert key.leaf is None

    def test_vector_mean_leaf(self) -> None:
        key = MetricKey.parse("breed/f1/test/mean")
        assert key.task == "breed"
        assert key.name == "f1"
        assert key.stage == Stage.TEST
        assert key.leaf == "mean"

    def test_vector_per_class_leaf(self) -> None:
        key = MetricKey.parse("breed/f1/test/Abyssinian")
        assert key.stage == Stage.TEST
        assert key.leaf == "Abyssinian"

    def test_two_part_key_has_no_stage(self) -> None:
        key = MetricKey.parse("label/accuracy")
        assert key.stage is None
        assert key.is_loss is False

    def test_unknown_third_segment_is_not_a_stage(self) -> None:
        key = MetricKey.parse("label/accuracy/epoch")
        assert key.stage is None


class TestDisplayName:
    """display_name strips both the stage and the vector leaf (summary rows)."""

    def test_loss_aggregate(self) -> None:
        assert MetricKey.parse("loss/val/total").display_name == "loss/total"

    def test_scalar_metric(self) -> None:
        assert MetricKey.parse("species/f1/val").display_name == "species/f1"

    def test_vector_mean(self) -> None:
        assert MetricKey.parse("breed/f1/test/mean").display_name == "breed/f1"

    def test_unclassified_returns_raw(self) -> None:
        assert MetricKey.parse("label/accuracy").display_name == "label/accuracy"


class TestWithoutStage:
    """without_stage removes only the stage segment, keeping any leaf (progress bar)."""

    def test_metric(self) -> None:
        assert MetricKey.parse("label/accuracy/train").without_stage() == "label/accuracy"

    def test_loss(self) -> None:
        assert MetricKey.parse("loss/val/total").without_stage() == "loss/total"

    def test_metric_keeps_leaf(self) -> None:
        assert MetricKey.parse("breed/f1/test/Abyssinian").without_stage() == "breed/f1/Abyssinian"

    def test_no_stage_unchanged(self) -> None:
        assert MetricKey.parse("label/accuracy").without_stage() == "label/accuracy"


class TestWithoutLeaf:
    """without_leaf drops only the trailing vector leaf, keeping the stage (progress bar row)."""

    def test_vector_mean_drops_leaf(self) -> None:
        assert MetricKey.parse("breed/f1/test/mean").without_leaf() == "breed/f1/test"

    def test_scalar_metric_unchanged(self) -> None:
        assert MetricKey.parse("seg/iou/val").without_leaf() == "seg/iou/val"

    def test_per_class_leaf_dropped(self) -> None:
        assert MetricKey.parse("breed/f1/test/Abyssinian").without_leaf() == "breed/f1/test"
