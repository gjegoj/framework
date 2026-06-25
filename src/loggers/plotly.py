"""Plotly figure builders for backend-agnostic ``Plot`` DTOs.

This is the ONLY module in the framework that imports Plotly. All figure
construction lives here, behind the ``plot_builders`` registry. Adding a new
plot type requires only:

1. A new ``Plot`` dataclass in ``src/core/plotting.py``.
2. A builder function registered here under ``NewType.__name__``.

The ``build_plot`` dispatcher is the single entry point for callers
(e.g. ``ClearMLLogger.log_plot``).
"""

from __future__ import annotations

from collections.abc import Callable

import plotly.graph_objects as go

from src.core.plotting import BoxPlot, Plot
from src.core.registry import Registry

# Registry mapping Plot subtype name → builder function.
# Mirrors ``head_builders`` / ``callback_builders``.
# The value type is Callable[[Plot], go.Figure]; ``get(key)(plot)`` is the call pattern
# (distinct from ``create(key, plot)`` which would return the Callable itself given the typing).
plot_builders: Registry[Callable[[Plot], go.Figure]] = Registry("plot_builder")


def build_box_plot(plot: BoxPlot) -> go.Figure:
    """Build a Plotly box figure from a ``BoxPlot`` DTO.

    One ``go.Box`` trace is emitted per category, built entirely from
    precomputed quartiles (no raw data points). A scatter mean-marker trace
    overlays each box so the arithmetic mean is visible at a glance.

    ``go.Box`` field mapping from ``BoxStats``:

    +-----------------------+-------------------+
    | Plotly field          | BoxStats field    |
    +=======================+===================+
    | ``lowerfence``        | ``minimum``       |
    | ``q1``                | ``q25``           |
    | ``median``            | ``median``        |
    | ``q3``                | ``q75``           |
    | ``upperfence``        | ``maximum``       |
    | scatter ``y``         | ``mean``          |
    +-----------------------+-------------------+

    Parameters:
        plot (BoxPlot): Backend-agnostic box-plot descriptor.

    Returns:
        go.Figure: Plotly figure with one box trace + one mean-marker trace
        per category.
    """
    traces: list[go.BaseTraceType] = []

    for category, stats in zip(plot.categories, plot.boxes, strict=True):
        traces.append(
            go.Box(
                name=category,
                x=[category],  # pin the box to its category, else Plotly stacks all boxes at index 0
                lowerfence=[stats.minimum],
                q1=[stats.q25],
                median=[stats.median],
                q3=[stats.q75],
                upperfence=[stats.maximum],
                boxpoints=False,
            )
        )
        traces.append(
            go.Scatter(
                name=f"{category} mean",
                x=[category],
                y=[stats.mean],
                mode="markers",
                marker={"symbol": "x", "size": 5},
                showlegend=True,
            )
        )

    figure = go.Figure(data=traces)
    # ``overlay``, not ``group``: each stage is its own single-box trace at a distinct
    # category, so grouping would slot each trace into a side column and slide its box off
    # the category label (away from the mean marker). Overlay centers each box on its stage.
    #
    # Legend on, horizontal, anchored just above the plot — a default bottom legend gets
    # cropped by ClearML's short inline preview, so it would only appear in the full-screen
    # view; a top row stays inside the always-rendered area.
    figure.update_layout(
        title=plot.title,
        yaxis_title=plot.y_label,
        boxmode="overlay",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    return figure


def build_plot(plot: Plot) -> go.Figure:
    """Dispatch a ``Plot`` DTO to the registered builder and return the figure.

    Uses ``type(plot).__name__`` as the registry key, so every concrete ``Plot``
    subtype must have a builder registered in ``plot_builders``.

    Parameters:
        plot (Plot): Backend-agnostic plot descriptor to render.

    Returns:
        go.Figure: The rendered Plotly figure.

    Raises:
        KeyError: If no builder is registered for ``type(plot).__name__``.
    """
    return plot_builders.get(type(plot).__name__)(plot)


# Registered after definition (not via @decorator) so mypy retains build_box_plot's
# precise BoxPlot parameter type for direct callers — the @decorator form would widen
# the bound name to the registry's Callable[[Plot], go.Figure] value type, erasing it.
plot_builders.register(BoxPlot.__name__)(build_box_plot)

__all__ = ["build_box_plot", "build_plot", "plot_builders"]
