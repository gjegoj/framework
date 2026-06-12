"""Visualization service: render predictions vs ground truth to interactive HTML.

Pipeline: a producer callback builds a ``SampleView`` per example (image +
FiftyOne-style ``Label`` fields) via per-task ``Annotator`` strategies, a
``Renderer`` turns the IR into a self-contained HTML document, and a ``PlotLogger``
ships it. Both axes of variation are registries (``annotators`` by task axes,
``label_renderers`` by Label type), so new task types extend by addition.
"""

from src.visualization.entities import Classification, Classifications, Label, SampleView
from src.visualization.pipeline import build_sample_views
from src.visualization.renderer import PlotlyRenderer, Renderer

__all__ = [
    "Classification",
    "Classifications",
    "Label",
    "PlotlyRenderer",
    "Renderer",
    "SampleView",
    "build_sample_views",
]
