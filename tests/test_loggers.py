"""Unit tests for the ClearML logger adapter's metric-name → (title, series) split.

``_split_metric_name`` is a pure static method; it needs no ClearML Task (the
``clearml`` import is lazy, inside ``__init__``), so these run without the optional
dependency installed.
"""

from __future__ import annotations

import pytest

from src.loggers.clearml import ClearMLLogger

split = ClearMLLogger._split_metric_name


class TestSplitMetricName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            # Losses: stage in the middle → pulled out as the series so train/val/test
            # of the same loss share one title (graph).
            ("loss/train/total", ("loss/total", "train")),
            ("loss/val/total", ("loss/total", "val")),
            ("loss/test/total", ("loss/total", "test")),
            # Composite-loss components keep a per-component graph, grouped by stage.
            ("loss/train/mask/cross_entropy", ("loss/mask/cross_entropy", "train")),
            ("loss/val/mask/cross_entropy", ("loss/mask/cross_entropy", "val")),
            # Metrics are UNCHANGED (last segment = series) — losses are the only exception.
            # Averaged metric: stage trails → train/val/test group on one graph.
            ("species/f1/val", ("species/f1", "val")),
            ("species/f1/train", ("species/f1", "train")),
            # Per-class metric: class name trails → all classes on one per-stage graph
            # (this is the case that must NOT be regrouped by stage).
            ("breed/f1/train/Abyssinian", ("breed/f1/train", "Abyssinian")),
            ("breed/f1/train/Bengal", ("breed/f1/train", "Bengal")),
            # A task literally containing a stage word in a non-loss key stays default.
            ("foo/bar/baz", ("foo/bar", "baz")),
            # Single segment → "value" series.
            ("lr", ("lr", "value")),
        ],
    )
    def test_split(self, name: str, expected: tuple[str, str]) -> None:
        assert split(name) == expected

    def test_metrics_are_not_regrouped_by_stage(self) -> None:
        """Per-class metrics for a stage stay on ONE graph (the regression we fixed)."""
        titles = {split(f"breed/f1/train/{cls}")[0] for cls in ("Abyssinian", "Bengal", "Birman")}
        assert titles == {"breed/f1/train"}

    def test_train_and_val_share_one_title(self) -> None:
        """The whole point: a loss's train/val/test land on the same graph (title)."""
        titles = {split(f"loss/{stage}/total")[0] for stage in ("train", "val", "test")}
        assert titles == {"loss/total"}
