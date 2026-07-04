"""Tests for the Plotly figure builders in ``src/loggers/plotly.py``.

Verifies:
- ``build_box_plot`` builds a ``go.Figure`` with one box trace per category.
- ``build_plot`` dispatches a ``BoxPlot`` to ``build_box_plot`` by type name.
- ``build_plot`` raises a ``KeyError`` for an unregistered ``Plot``-shaped object.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import pytest

from src.core.plotting import BoxPlot, BoxStats
from src.loggers.plotly import build_box_plot, build_plot


def _make_box_stats(offset: float = 0.0) -> BoxStats:
    """Return a ``BoxStats`` with distinct, valid quartile values."""
    return BoxStats(
        minimum=0.0 + offset,
        q25=1.0 + offset,
        median=2.0 + offset,
        q75=3.0 + offset,
        maximum=4.0 + offset,
        mean=2.0 + offset,
    )


class TestBuildBoxPlot:
    def test_returns_go_figure(self) -> None:
        plot = BoxPlot(
            title="Test plot",
            categories=["train"],
            boxes=[_make_box_stats()],
        )
        figure = build_box_plot(plot)
        assert isinstance(figure, go.Figure)

    def test_box_trace_count_equals_category_count(self) -> None:
        """One go.Box trace per category (plus one scatter mean-marker per category)."""
        categories = ["train", "val", "test"]
        plot = BoxPlot(
            title="Three stages",
            categories=categories,
            boxes=[_make_box_stats(i) for i in range(len(categories))],
        )
        figure = build_box_plot(plot)
        box_traces = [trace for trace in figure.data if trace.type == "box"]
        assert len(box_traces) == len(categories)

    def test_box_trace_names_match_categories(self) -> None:
        categories = ["train", "val"]
        plot = BoxPlot(
            title="Stage boxes",
            categories=categories,
            boxes=[_make_box_stats(i) for i in range(len(categories))],
        )
        figure = build_box_plot(plot)
        box_names = [trace.name for trace in figure.data if trace.type == "box"]
        assert box_names == categories

    def test_mean_marker_traces_present(self) -> None:
        """One scatter mean-marker trace per category."""
        categories = ["train", "val"]
        plot = BoxPlot(
            title="Mean markers",
            categories=categories,
            boxes=[_make_box_stats(i) for i in range(len(categories))],
        )
        figure = build_box_plot(plot)
        scatter_traces = [trace for trace in figure.data if trace.type == "scatter"]
        assert len(scatter_traces) == len(categories)

    def test_legend_is_enabled_and_horizontal_on_top(self) -> None:
        """Legend is explicitly on and laid out as a horizontal row above the plot, so a
        short ClearML preview (which crops a default bottom legend) still shows it."""
        plot = BoxPlot(title="Legend", categories=["train"], boxes=[_make_box_stats()])
        figure = build_box_plot(plot)
        assert figure.layout.showlegend is True
        assert figure.layout.legend.orientation == "h"

    def test_y_label_applied(self) -> None:
        plot = BoxPlot(
            title="Custom y",
            categories=["train"],
            boxes=[_make_box_stats()],
            y_label="score",
        )
        figure = build_box_plot(plot)
        assert figure.layout.yaxis.title.text == "score"

    def test_boxes_overlay_not_grouped(self) -> None:
        """boxmode must be 'overlay': each stage is its own single-box trace at its own
        category, so 'group' would offset each trace into a side sub-slot, sliding the
        boxes off their category labels (away from the mean markers)."""
        plot = BoxPlot(
            title="Overlay",
            categories=["train", "val", "test"],
            boxes=[_make_box_stats(index) for index in range(3)],
        )
        figure = build_box_plot(plot)
        assert figure.layout.boxmode == "overlay"

    def test_box_trace_x_positions_match_categories(self) -> None:
        """Each box must carry its category as ``x``; otherwise Plotly piles every box at
        one implicit category while the mean markers (which do set ``x``) sit elsewhere."""
        categories = ["train", "val", "test"]
        plot = BoxPlot(
            title="Placed boxes",
            categories=categories,
            boxes=[_make_box_stats(i) for i in range(len(categories))],
        )
        figure = build_box_plot(plot)
        box_x = [list(trace.x) for trace in figure.data if trace.type == "box"]
        assert box_x == [[category] for category in categories]

    def test_quartile_fields_set_on_box_trace(self) -> None:
        stats = _make_box_stats()
        plot = BoxPlot(title="Q fields", categories=["train"], boxes=[stats])
        figure = build_box_plot(plot)
        box_trace = next(trace for trace in figure.data if trace.type == "box")
        assert list(box_trace.lowerfence) == [stats.minimum]
        assert list(box_trace.q1) == [stats.q25]
        assert list(box_trace.median) == [stats.median]
        assert list(box_trace.q3) == [stats.q75]
        assert list(box_trace.upperfence) == [stats.maximum]


class TestBuildPlot:
    def test_dispatches_box_plot(self) -> None:
        plot = BoxPlot(
            title="Dispatch test",
            categories=["train", "val"],
            boxes=[_make_box_stats(i) for i in range(2)],
        )
        figure = build_plot(plot)
        assert isinstance(figure, go.Figure)
        box_traces = [trace for trace in figure.data if trace.type == "box"]
        assert len(box_traces) == 2

    def test_unknown_plot_type_raises_key_error(self) -> None:
        """An unregistered Plot-shaped object must raise KeyError from the registry."""

        @dataclass(frozen=True, slots=True)
        class UnknownPlot:
            title: str = "unknown"

        with pytest.raises(KeyError, match="plot_builder"):
            build_plot(UnknownPlot())  # type: ignore[arg-type]
