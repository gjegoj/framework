"""Tests for the MetricKey value object — the single parser of logged-scalar keys.

The grammar (``{task}/{metric}/{stage}[/leaf]`` for task metrics,
``loss/{stage}/{component}`` for losses) is parsed in exactly one place; these
tests pin every shape the live consumers (summary / progress bar / ClearML)
depend on.
"""

from __future__ import annotations

import pytest

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

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("loss/val/total", "loss/total", id="loss_aggregate"),
            pytest.param("species/f1/val", "species/f1", id="scalar_metric"),
            pytest.param("breed/f1/test/mean", "breed/f1", id="vector_mean"),
            pytest.param("label/accuracy", "label/accuracy", id="unclassified_returns_raw"),
        ],
    )
    def test_display_name(self, key: str, expected: str) -> None:
        assert MetricKey.parse(key).display_name == expected


class TestWithoutStage:
    """without_stage removes only the stage segment, keeping any leaf (progress bar)."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("label/accuracy/train", "label/accuracy", id="metric"),
            pytest.param("loss/val/total", "loss/total", id="loss"),
            pytest.param("breed/f1/test/Abyssinian", "breed/f1/Abyssinian", id="metric_keeps_leaf"),
            pytest.param("label/accuracy", "label/accuracy", id="no_stage_unchanged"),
        ],
    )
    def test_without_stage(self, key: str, expected: str) -> None:
        assert MetricKey.parse(key).without_stage() == expected


class TestWithoutLeaf:
    """without_leaf drops only the trailing vector leaf, keeping the stage (progress bar row)."""

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            pytest.param("breed/f1/test/mean", "breed/f1/test", id="vector_mean_drops_leaf"),
            pytest.param("seg/iou/val", "seg/iou/val", id="scalar_metric_unchanged"),
            pytest.param("breed/f1/test/Abyssinian", "breed/f1/test", id="per_class_leaf_dropped"),
        ],
    )
    def test_without_leaf(self, key: str, expected: str) -> None:
        assert MetricKey.parse(key).without_leaf() == expected
