"""Shared test doubles (fakes) — pure classes, no pytest fixtures.

``FakePlotLogger`` is the full-contract artifact-logger double consumed by the metrics,
callbacks, reporting and sample-log tests; it records every call for assertion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from src.core.plotting import Plot
from src.core.ports import (
    CurveLogger,
    HistogramLogger,
    HtmlLogger,
    MatrixLogger,
    PlotLogger,
    SingleValueLogger,
)


class FakePlotLogger(MatrixLogger, CurveLogger, HtmlLogger, SingleValueLogger, HistogramLogger, PlotLogger):
    """Full-contract test double recording every artifact-logger call (matrix/curve/html/single_value/histogram/plot)."""

    def __init__(self) -> None:
        self.matrix_calls: list[dict[str, Any]] = []
        self.curve_calls: list[dict[str, Any]] = []
        self.html_calls: list[dict[str, Any]] = []
        self.single_values: dict[str, float] = {}
        self.histogram_calls: list[dict[str, Any]] = []
        self.plot_calls: list[Plot] = []

    def log_matrix(
        self,
        title: str,
        matrix: torch.Tensor,
        iteration: int,
        labels: list[str] | None = None,
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        self.matrix_calls.append(
            {"title": title, "matrix": matrix, "iteration": iteration, "labels": labels, "xaxis": xaxis, "yaxis": yaxis}
        )

    def log_curve(
        self,
        title: str,
        x: torch.Tensor,
        y: torch.Tensor,
        iteration: int,
        series: str = "curve",
        xaxis: str | None = None,
        yaxis: str | None = None,
    ) -> None:
        self.curve_calls.append(
            {"title": title, "x": x, "y": y, "iteration": iteration, "series": series, "xaxis": xaxis, "yaxis": yaxis}
        )

    def log_html(self, title: str, html: str, iteration: int) -> None:
        self.html_calls.append({"title": title, "html": html, "iteration": iteration})

    def log_single_value(self, name: str, value: float) -> None:
        self.single_values[name] = value

    def log_histogram(self, title: str, series: str, values: Sequence[float], labels: list[str] | None = None) -> None:
        self.histogram_calls.append({"title": title, "series": series, "values": list(values), "labels": labels})

    def log_plot(self, plot: Plot) -> None:
        self.plot_calls.append(plot)
