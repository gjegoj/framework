"""Unit tests for the metrics progress bar.

These exercise the pure helpers and the bookkeeping methods directly,
without a Lightning trainer or a live ``Live`` display.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.callbacks.progress_bar import (
    MetricsProgressBar,
    _mode_from_flag,
    _split_stage,
)

# ---------------------------------------------------------------- fakes / builders


class _FakeDirectionProvider:
    """Structurally satisfies ``MetricDirectionProvider`` for binding tests."""

    def __init__(self, directions: dict[str, bool | None]) -> None:
        self._directions = directions

    def metric_directions(self) -> dict[str, bool | None]:
        return self._directions


def _bar(**kwargs: Any) -> MetricsProgressBar:
    """Construct a bar without starting any Live display."""
    return MetricsProgressBar(**kwargs)


# ---------------------------------------------------------------- _mode_from_flag


class TestModeFromFlag:
    def test_higher_is_better_is_max(self) -> None:
        assert _mode_from_flag(True) == "max"

    def test_lower_is_better_is_min(self) -> None:
        assert _mode_from_flag(False) == "min"

    def test_no_direction_is_none(self) -> None:
        assert _mode_from_flag(None) is None


# ---------------------------------------------------------------- _split_stage


class TestSplitStage:
    def test_extracts_train_stage(self) -> None:
        assert _split_stage("label/accuracy/train") == ("label/accuracy", "train")

    def test_extracts_val_stage(self) -> None:
        assert _split_stage("seg/iou/val") == ("seg/iou", "val")

    def test_extracts_test_stage(self) -> None:
        assert _split_stage("label/accuracy/test") == ("label/accuracy", "test")

    def test_stage_in_the_middle(self) -> None:
        assert _split_stage("loss/val/total") == ("loss/total", "val")

    def test_no_stage_returns_name_and_none(self) -> None:
        assert _split_stage("label/accuracy") == ("label/accuracy", None)


# ---------------------------------------------------------------- _bind_directions


class TestBindDirections:
    def test_translates_provider_flags_to_modes(self) -> None:
        provider = _FakeDirectionProvider(
            {"label/accuracy/train": True, "label/mse/train": False, "label/cm/train": None}
        )
        bar = _bar()
        bar._bind_directions(provider)
        assert bar._direction_by_key == {
            "label/accuracy/train": "max",
            "label/mse/train": "min",
            "label/cm/train": None,
        }

    def test_non_provider_binds_nothing(self) -> None:
        bar = _bar()
        bar._bind_directions(object())
        assert bar._direction_by_key == {}


# ---------------------------------------------------------------- _direction_for


class TestDirectionFor:
    def test_reads_from_bound_map(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        assert bar._direction_for("label/accuracy/val") == "max"

    def test_loss_namespace_defaults_to_min(self) -> None:
        bar = _bar(loss_key="loss")
        assert bar._direction_for("loss/train/total") == "min"

    def test_unbound_non_loss_has_no_direction(self) -> None:
        bar = _bar()
        assert bar._direction_for("label/perplexity/val") is None


# ---------------------------------------------------------------- _displayed_key


class TestDisplayedKey:
    def test_loss_total_passes_through(self) -> None:
        bar = _bar(loss_key="loss", metric_filters=None)
        assert bar._displayed_key("loss/train/total") == "loss/train/total"

    def test_loss_component_is_skipped(self) -> None:
        """Only the 3-part loss total is shown, not 4-part per-component losses."""
        bar = _bar(loss_key="loss")
        assert bar._displayed_key("loss/train/breed/cross_entropy") is None

    def test_three_part_scalar_kept(self) -> None:
        bar = _bar()
        assert bar._displayed_key("label/accuracy/val") == "label/accuracy/val"

    def test_test_stage_kept(self) -> None:
        bar = _bar()
        assert bar._displayed_key("label/accuracy/test") == "label/accuracy/test"

    def test_vector_mean_collapses_to_three_parts(self) -> None:
        """A vector metric's ``.../mean`` aggregate becomes the plain task/metric/stage row."""
        bar = _bar()
        assert bar._displayed_key("breed/f1/test/mean") == "breed/f1/test"
        assert bar._displayed_key("breed/recall/val/mean") == "breed/recall/val"

    def test_per_class_vector_leaf_is_skipped(self) -> None:
        bar = _bar()
        assert bar._displayed_key("breed/f1/test/Abyssinian") is None

    def test_non_stage_metric_skipped(self) -> None:
        bar = _bar()
        assert bar._displayed_key("label/accuracy/epoch") is None

    def test_two_part_metric_skipped(self) -> None:
        bar = _bar()
        assert bar._displayed_key("label/accuracy") is None

    def test_filters_restrict_to_substrings(self) -> None:
        bar = _bar(loss_key="", metric_filters=["iou"])
        assert bar._displayed_key("seg/iou/val") == "seg/iou/val"
        assert bar._displayed_key("label/accuracy/val") is None


# ---------------------------------------------------------------- _track


class TestTrack:
    def test_records_current_value(self) -> None:
        bar = _bar()
        bar._track("label/accuracy/val", 0.5)
        assert bar._current_values["label/accuracy/val"] == 0.5

    def test_step_delta_captures_change(self) -> None:
        bar = _bar()
        bar._track("label/accuracy/val", 0.5)
        bar._track("label/accuracy/val", 0.8)
        assert bar._step_deltas["label/accuracy/val"] == pytest.approx(0.3)

    def test_first_observation_sets_best_without_delta(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        bar._track("label/accuracy/val", 0.5)
        assert bar._best_values["label/accuracy/val"] == 0.5
        assert "label/accuracy/val" not in bar._best_deltas

    def test_max_metric_improves_on_higher_value(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        bar._track("label/accuracy/val", 0.5)
        bar._track("label/accuracy/val", 0.8)
        assert bar._best_values["label/accuracy/val"] == 0.8
        assert bar._best_deltas["label/accuracy/val"] == pytest.approx(0.3)

    def test_max_metric_keeps_best_on_lower_value(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        bar._track("label/accuracy/val", 0.8)
        bar._track("label/accuracy/val", 0.5)
        assert bar._best_values["label/accuracy/val"] == 0.8

    def test_min_metric_improves_on_lower_value(self) -> None:
        bar = _bar(loss_key="loss")
        bar._track("loss/train/total", 1.0)
        bar._track("loss/train/total", 0.3)
        assert bar._best_values["loss/train/total"] == 0.3
        assert bar._best_deltas["loss/train/total"] == pytest.approx(-0.7)

    def test_directionless_metric_has_no_best(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/confusion_matrix/val"] = None
        bar._track("label/confusion_matrix/val", 0.5)
        assert "label/confusion_matrix/val" not in bar._best_values


# ---------------------------------------------------------------- _format_cell


class TestFormatCell:
    def test_none_value_renders_dash(self) -> None:
        bar = _bar()
        assert bar._format_cell("label/accuracy/val", None, {}).plain == "-"

    def test_value_without_delta_has_no_arrow(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        cell = bar._format_cell("label/accuracy/val", 0.5, {})
        assert cell.plain == "0.5000"

    def test_improvement_for_max_metric_is_green_up_arrow(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        cell = bar._format_cell("label/accuracy/val", 0.8, {"label/accuracy/val": 0.3})
        assert cell.plain == "0.8000 ▲0.3000"
        assert any(span.style == "green" for span in cell.spans)

    def test_regression_for_max_metric_is_red(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/val"] = "max"
        cell = bar._format_cell("label/accuracy/val", 0.5, {"label/accuracy/val": -0.3})
        assert cell.plain == "0.5000 ▼0.3000"
        assert any(span.style == "red" for span in cell.spans)

    def test_improvement_for_min_metric_is_green_down_arrow(self) -> None:
        bar = _bar(loss_key="loss")
        cell = bar._format_cell("loss/train/total", 0.3, {"loss/train/total": -0.7})
        assert cell.plain == "0.3000 ▼0.7000"
        assert any(span.style == "green" for span in cell.spans)

    def test_directionless_metric_shows_value_only(self) -> None:
        bar = _bar()
        cell = bar._format_cell("label/perplexity/val", 0.5, {"label/perplexity/val": 0.1})
        assert cell.plain == "0.5000"


# ---------------------------------------------------------------- _build_table


class TestBuildTable:
    def test_row_per_base_metric_with_train_and_best_columns(self) -> None:
        bar = _bar()
        bar._direction_by_key["label/accuracy/train"] = "max"
        bar._track("label/accuracy/train", 0.6)
        bar._track("label/accuracy/train", 0.9)
        table = bar._build_table({"label/accuracy/train": 0.9})
        assert table.row_count == 1
        assert [column.header for column in table.columns] == [
            "Metric",
            "Train",
            "Best (train)",
            "Val",
            "Best (val)",
            "Test",
        ]

    def test_test_stage_value_is_rendered(self) -> None:
        """Test-stage metrics land in the Test column (regression: they used to be dropped)."""
        bar = _bar()
        table = bar._build_table({"label/accuracy/test": 0.88})
        assert table.row_count == 1
        # Cell content lives in the rendered column data; the Test column is the last one.
        test_column = table.columns[-1]
        assert test_column.header == "Test"
        assert any("0.88" in str(cell) for cell in test_column.cells)

    def test_vector_mean_renders_one_row_named_by_base(self) -> None:
        """A collapsed vector-mean key (task/metric/stage) renders a single task/metric row."""
        bar = _bar()
        table = bar._build_table({"breed/f1/test": 0.158})
        assert table.row_count == 1
        assert any("breed/f1" in str(cell) for cell in table.columns[0].cells)
        assert any("0.158" in str(cell) for cell in table.columns[-1].cells)


# ---------------------------------------------------------------- on_test_end


class TestOnTestEnd:
    def test_refreshes_so_test_metrics_reach_the_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RichProgressBar has no post-test refresh; our override must force one."""
        bar = _bar()
        refreshed: list[bool] = []
        monkeypatch.setattr(bar, "refresh", lambda hard=False: refreshed.append(hard))
        bar.on_test_end(trainer=object(), pl_module=object())
        assert refreshed == [True]


# ---------------------------------------------------------------- teardown


class TestTeardown:
    def test_stops_live_and_prints_separator(self) -> None:
        """The Live group is stopped and a blank line printed, so output isn't glued to the table."""
        from unittest.mock import MagicMock

        bar = _bar()
        bar._console = MagicMock()
        bar._live = MagicMock()
        live = bar._live
        bar.teardown(MagicMock(), MagicMock(), "fit")
        live.stop.assert_called_once_with()
        bar._console.print.assert_called_once_with()
        assert bar._live is None

    def test_noop_without_live(self) -> None:
        from unittest.mock import MagicMock

        bar = _bar()
        bar._console = MagicMock()
        bar.teardown(MagicMock(), MagicMock(), "fit")  # _live is None → nothing printed, no crash
        bar._console.print.assert_not_called()
