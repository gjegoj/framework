"""Renderer: turn ``SampleView`` IR into a self-contained interactive HTML grid.

The render side of the pipeline; Plotly lives here and nowhere else (imported
lazily so the framework does not hard-require it). Per-Label rendering is
dispatched through the ``label_renderers`` registry keyed by Label type name — the
OCP seam mirroring ``annotators``. Phase 1 ships text captions for classification
labels; later phases register renderers for new Label types (e.g. mask overlays).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.registry import Registry
from src.visualization.entities import Classification, Classifications, Label, SampleView

label_renderers: Registry[LabelRenderer] = Registry("label_renderer")


class LabelRenderer(ABC):
    """Produces a human-readable caption for one ``Label`` type."""

    @abstractmethod
    def caption(self, label: Label) -> str:
        """Return a short badge string for ``label`` (e.g. ``"cat (0.91)"``)."""


@label_renderers.register(Classification.__name__)
class ClassificationLabelRenderer(LabelRenderer):
    def caption(self, label: Label) -> str:
        assert isinstance(label, Classification)
        if label.confidence is None:
            return label.label
        return f"{label.label} ({label.confidence:.2f})"


@label_renderers.register(Classifications.__name__)
class ClassificationsLabelRenderer(LabelRenderer):
    def caption(self, label: Label) -> str:
        assert isinstance(label, Classifications)
        if not label.items:
            return "∅"
        parts = [
            item.label if item.confidence is None else f"{item.label} ({item.confidence:.2f})" for item in label.items
        ]
        return ", ".join(parts)


def _caption_label(label: Label) -> str:
    return label_renderers.create(type(label).__name__).caption(label)


_CAPTION_MAX_CHARS = 56
_LINE_PX = 14
_TITLE_BLOCK_PX = 44
_CELL_IMAGE_PX = 220


def _axis_domain(axis_id: int, axis: str) -> str:
    return f"{axis} domain" if axis_id == 1 else f"{axis}{axis_id} domain"


def _caption_line_count(caption: str) -> int:
    return caption.count("<br>") + 1 if caption else 1


def _truncate_caption(text: str, limit: int = _CAPTION_MAX_CHARS) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


class Renderer(ABC):
    """Turns a list of ``SampleView`` into one self-contained HTML document."""

    @abstractmethod
    def render(self, samples: list[SampleView], title: str) -> str:
        """Render ``samples`` into a self-contained interactive HTML string."""


class PlotlyRenderer(Renderer):
    """Grid of image cells with gt/pred badge titles, coloured by correctness.

    Each ``SampleView`` becomes a subplot showing its image; the subplot title is
    composed from the sample's task fields (``{task}_gt`` / ``{task}_pred``) and is
    coloured red when any task tag is ``wrong``. The output is a self-contained
    interactive Plotly document (zoom/pan/hover), embeddable as ClearML media.

    Parameters:
        max_columns (int): Maximum grid columns.
        include_plotlyjs (str): Passed to ``Figure.to_html`` — ``"cdn"`` (light,
            needs internet in the viewer) or ``"inline"`` (fully offline, heavier).
    """

    def __init__(self, max_columns: int = 4, include_plotlyjs: str = "cdn") -> None:
        self._max_columns = max_columns
        self._include_plotlyjs = include_plotlyjs

    def render(self, samples: list[SampleView], title: str) -> str:

        if not samples:
            return f"<html><body><p>{title}: no samples</p></body></html>"

        columns = min(self._max_columns, len(samples))
        rows = math.ceil(len(samples) / columns)
        captions = [self._sample_caption(sample) for sample in samples]
        max_lines = max(_caption_line_count(caption) for caption in captions)
        caption_band = max_lines * _LINE_PX + 10
        vertical_spacing = (0.14 + 0.04 * max_lines) if rows > 1 else 0.02

        figure = make_subplots(
            rows=rows,
            cols=columns,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.06,
        )

        for position, sample in enumerate(samples):
            row = position // columns + 1
            col = position % columns + 1
            axis_id = (row - 1) * columns + col
            figure.add_trace(go.Image(z=sample.image, hoverinfo="skip"), row=row, col=col)

            wrong = any(tag.endswith(":wrong") for tag in sample.tags)
            figure.add_annotation(
                text=captions[position],
                xref=_axis_domain(axis_id, "x"),
                yref=_axis_domain(axis_id, "y"),
                x=0.5,
                y=-0.04,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font={"size": 9, "color": "#c0392b" if wrong else "#2c3e50"},
                align="center",
            )

        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)

        row_height = _CELL_IMAGE_PX + caption_band
        gap_px = int(vertical_spacing * _CELL_IMAGE_PX) if rows > 1 else 0
        figure_height = _TITLE_BLOCK_PX + rows * row_height + max(0, rows - 1) * gap_px
        figure.update_layout(
            title=title,
            height=figure_height,
            margin={"t": _TITLE_BLOCK_PX, "l": 10, "r": 10, "b": caption_band},
        )

        html: str = figure.to_html(include_plotlyjs=self._include_plotlyjs, full_html=True)
        return html

    @staticmethod
    def _sample_caption(sample: SampleView) -> str:
        """Compose ``task: gt=… | pred=…`` lines from the sample's field pairs."""
        tasks = sorted({name.rsplit("_", 1)[0] for name in sample.fields if name.endswith(("_gt", "_pred"))})
        lines: list[str] = []
        for task_name in tasks:
            gt = sample.fields.get(f"{task_name}_gt")
            pred = sample.fields.get(f"{task_name}_pred")
            gt_text = _truncate_caption(_caption_label(gt) if gt is not None else "?")
            pred_text = _truncate_caption(_caption_label(pred) if pred is not None else "?")
            lines.append(f"{task_name}: gt={gt_text} | pred={pred_text}")
        return "<br>".join(lines)
