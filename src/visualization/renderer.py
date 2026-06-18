"""Renderer: turn SampleView IR into a self-contained interactive HTML document.

The render side of the visualization pipeline — no plotting library. Produces one
self-contained HTML/CSS/JS document: a responsive grid of image cells with overlays
(FiftyOne-style chips and segmentation mask layers) and a collapsible sidebar that
toggles overlay visibility. Images are inlined as base64 PNG; CSS/JS are inlined from
``assets/``. The output flows through ``PlotLogger.log_html`` and embeds in ClearML.

Per-Label rendering is dispatched through the ``label_renderers`` registry (OCP seam):
a ``LabelRenderer`` turns a ``Label`` into ``FieldItem``s — each a toggleable cell
overlay plus its sidebar entry, tagged with a ``zone`` (``"chips"`` flow-layout, or
``"cover"`` full-cell for masks). The ``HtmlRenderer`` is a layout shell: it resolves a
per-task color palette, places overlays by zone, and aggregates the sidebar, never
branching on label type. A new rendering peculiarity is a new ``LabelRenderer``; a new
render backend is a new ``Renderer``.
"""

from __future__ import annotations

import base64
import html
import io
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.registry import Registry
from src.visualization.colors import FALLBACK_COLOR, REGRESSION_COLOR, hex_to_rgb, task_palette
from src.visualization.entities import (
    Classification,
    Classifications,
    Label,
    Regression,
    SampleView,
    Segmentation,
)
from src.visualization.masks import mask_overlay_uri

KINDS: tuple[str, ...] = ("gt", "pred")
_MAX_CHIP_CHARS = 22

_ASSETS = Path(__file__).parent / "assets"
_GRID_CSS = (_ASSETS / "grid.css").read_text(encoding="utf-8")
_GRID_JS = (_ASSETS / "grid.js").read_text(encoding="utf-8")

# Static overlay scaffold; grid.js wires cell clicks to clone a cell into it (large).
_LIGHTBOX = (
    '<div class="lb hidden" id="lb"><div class="stage" id="lb-stage">'
    '<div id="lb-holder"></div>'
    '<button class="nav prev" id="lb-prev" title="Previous (←)">‹</button>'
    '<button class="nav next" id="lb-next" title="Next (→)">›</button>'
    '<button class="close" id="lb-close" title="Close (Esc)">✕</button>'
    '<div class="count" id="lb-count"></div>'
    "</div></div>"
)


def field_key(task: str, kind: str, leaf: str) -> str:
    """Build the join key tying a cell overlay to its sidebar checkbox."""
    return f"{task}::{kind}::{leaf}"


@dataclass(frozen=True)
class FieldContext:
    """The task/kind, chip budget, and resolved colors a ``Label`` is rendered under.

    Parameters:
        task (str): The owning task name.
        kind (str): ``"gt"`` or ``"pred"``.
        max_chip_chars (int): Chip display-text budget before truncation.
        colors (Mapping[str, str]): ``leaf -> hex`` for this task (the per-task palette).
    """

    task: str
    kind: str
    max_chip_chars: int = _MAX_CHIP_CHARS
    colors: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldItem:
    """One leaf of a rendered field: a toggleable cell overlay + its sidebar entry.

    Parameters:
        data_key (str): ``task::kind::leaf`` — joins the overlay to its checkbox.
        overlay_html (str): Markup placed in the cell (a chip ``<span>`` or a mask ``<img>``).
        zone (str): ``"chips"`` (flow layout) or ``"cover"`` (full-cell, e.g. masks).
        sidebar_label (str): Leaf text shown in the sidebar.
        color (str): Swatch / chip / mask color.
        filled (bool): gt → filled swatch+chip / solid mask border; pred → outlined / dashed.
    """

    data_key: str
    overlay_html: str
    zone: str
    sidebar_label: str
    color: str
    filled: bool


def render_chip(
    data_key: str,
    text: str,
    color: str,
    *,
    filled: bool,
    title: str | None = None,
    max_chars: int = _MAX_CHIP_CHARS,
) -> str:
    """Render one chip ``<span>`` (the shared builder for every chip-shaped label)."""
    shown = text if len(text) <= max_chars else text[: max_chars - 1] + "…"
    css_kind = "gt" if filled else "pred"
    style = f"background:{color}" if filled else f"border-color:{color};color:{color}"
    return (
        f'<span class="layer chip {css_kind}" data-key="{html.escape(data_key, quote=True)}" '
        f'style="{style}" title="{html.escape(title or text, quote=True)}">{html.escape(shown)}</span>'
    )


label_renderers: Registry[LabelRenderer] = Registry("label_renderer")


class LabelRenderer(ABC):
    """Renders one ``Label`` field (gt or pred) into its ``FieldItem``s."""

    @abstractmethod
    def render_field(self, context: FieldContext, label: Label) -> list[FieldItem]:
        """Return one ``FieldItem`` per leaf (class / component) of ``label``."""

    @abstractmethod
    def leaves(self, label: Label) -> list[str]:
        """Return the leaf names ``label`` contributes (without color) for palette sizing."""


def _chip_item(context: FieldContext, leaf: str, text: str, color: str) -> FieldItem:
    """Build a chip ``FieldItem`` for one leaf under ``context`` (gt filled / pred outlined)."""
    filled = context.kind == "gt"
    key = field_key(context.task, context.kind, leaf)
    overlay = render_chip(key, text, color, filled=filled, max_chars=context.max_chip_chars)
    return FieldItem(key, overlay, "chips", leaf, color, filled)


def _classification_text(item: Classification) -> str:
    return item.label if item.confidence is None else f"{item.label} {item.confidence:.2f}"


@label_renderers.register(Classification.__name__)
class ClassificationLabelRenderer(LabelRenderer):
    def render_field(self, context: FieldContext, label: Label) -> list[FieldItem]:
        assert isinstance(label, Classification)
        color = context.colors.get(label.label, FALLBACK_COLOR)
        return [_chip_item(context, label.label, _classification_text(label), color)]

    def leaves(self, label: Label) -> list[str]:
        assert isinstance(label, Classification)
        return [label.label]


@label_renderers.register(Classifications.__name__)
class ClassificationsLabelRenderer(LabelRenderer):
    def render_field(self, context: FieldContext, label: Label) -> list[FieldItem]:
        assert isinstance(label, Classifications)
        return [
            _chip_item(
                context,
                classification.label,
                _classification_text(classification),
                context.colors.get(classification.label, FALLBACK_COLOR),
            )
            for classification in label.items
        ]

    def leaves(self, label: Label) -> list[str]:
        assert isinstance(label, Classifications)
        return [classification.label for classification in label.items]


@label_renderers.register(Regression.__name__)
class RegressionLabelRenderer(LabelRenderer):
    """Regression: one neutral-colored chip per component; pred shows signed Δ as text."""

    def render_field(self, context: FieldContext, label: Label) -> list[FieldItem]:
        assert isinstance(label, Regression)
        items: list[FieldItem] = []
        for component in label.components:
            leaf = component.name or "value"
            number = f"{component.value:.2f}"
            body = f"{component.name} {number}" if component.name else number
            text = body if (context.kind == "gt" or component.error is None) else f"{body} Δ{component.error:+.2f}"
            items.append(_chip_item(context, leaf, text, REGRESSION_COLOR))
        return items

    def leaves(self, label: Label) -> list[str]:
        assert isinstance(label, Regression)
        return [component.name or "value" for component in label.components]


@label_renderers.register(Segmentation.__name__)
class SegmentationLabelRenderer(LabelRenderer):
    """Segmentation: one full-cell mask layer per class; gt solid / pred dashed border."""

    def render_field(self, context: FieldContext, label: Label) -> list[FieldItem]:
        assert isinstance(label, Segmentation)
        filled = context.kind == "gt"
        items: list[FieldItem] = []
        for seg_class in label.classes:
            color = context.colors.get(seg_class.name, FALLBACK_COLOR)
            key = field_key(context.task, context.kind, seg_class.name)
            overlay_uri = mask_overlay_uri(seg_class.mask, hex_to_rgb(color))
            overlay = f'<img class="layer mask" data-key="{html.escape(key, quote=True)}" src="{overlay_uri}">'
            items.append(FieldItem(key, overlay, "cover", seg_class.name, color, filled))
        return items

    def leaves(self, label: Label) -> list[str]:
        assert isinstance(label, Segmentation)
        return [seg_class.name for seg_class in label.classes]


def _render_field(context: FieldContext, label: Label) -> list[FieldItem]:
    return label_renderers.create(type(label).__name__).render_field(context, label)


def _leaves(label: Label) -> list[str]:
    return label_renderers.create(type(label).__name__).leaves(label)


def _split_field(name: str) -> tuple[str, str]:
    """Split a ``{task}_{kind}`` field name into ``(task, kind)``."""
    task, _, kind = name.rpartition("_")
    return task, kind


def _encode_image(image: np.ndarray) -> str:
    """Encode an ``[H, W, C]`` uint8 RGB array as a base64 PNG data URI."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class Renderer(ABC):
    """Turns a list of ``SampleView`` into one self-contained HTML document."""

    @abstractmethod
    def render(self, samples: list[SampleView], title: str) -> str:
        """Render ``samples`` into a self-contained interactive HTML string."""


class HtmlRenderer(Renderer):
    """Render SampleViews to a self-contained HTML/CSS/JS grid with a toggle sidebar.

    A layout shell: it resolves a per-task color palette from the classes present, asks
    each field's ``LabelRenderer`` for ``FieldItem``s, places each overlay in its ``zone``
    container, and aggregates the sidebar from the items — it never branches on label type.

    Parameters:
        max_chip_chars (int): Chip display-text budget before truncation (full text in
            the chip ``title`` tooltip).
    """

    def __init__(self, max_chip_chars: int = _MAX_CHIP_CHARS) -> None:
        self._max_chip_chars = max_chip_chars

    def render(self, samples: list[SampleView], title: str) -> str:
        safe_title = html.escape(title)
        palettes = _build_palettes(samples)
        taxonomy: dict[tuple[str, str], dict[str, FieldItem]] = {}
        cells: list[str] = []
        for sample in samples:
            cell_html, items = self._render_cell(sample, palettes)
            cells.append(cell_html)
            for task, kind, item in items:
                taxonomy.setdefault((task, kind), {})[item.data_key] = item
        sidebar = self._render_sidebar(taxonomy)
        return (
            "<!DOCTYPE html>\n"
            '<html><head><meta charset="utf-8">'
            f"<title>{safe_title}</title><style>{_GRID_CSS}</style></head>\n"
            "<body>\n"
            f'  <div class="sidebar"><h2>fields</h2>{sidebar}</div>\n'
            f'  <div class="main"><h2>{safe_title}</h2><div class="grid">{chr(10).join(cells)}</div></div>\n'
            f"  {_LIGHTBOX}\n"
            f"  <script>{_GRID_JS}</script>\n"
            "</body></html>"
        )

    def _render_cell(
        self, sample: SampleView, palettes: dict[str, dict[str, str]]
    ) -> tuple[str, list[tuple[str, str, FieldItem]]]:
        zones: dict[str, list[str]] = {"cover": [], "chips": []}
        collected: list[tuple[str, str, FieldItem]] = []
        for field_name, label in sample.fields.items():
            task, kind = _split_field(field_name)
            context = FieldContext(task, kind, self._max_chip_chars, colors=palettes.get(task, {}))
            for item in _render_field(context, label):
                zones[item.zone].append(item.overlay_html)
                collected.append((task, kind, item))
        image_uri = _encode_image(sample.image)
        cell_html = (
            f'<div class="cell"><img src="{image_uri}">'
            f'<div class="cover">{"".join(zones["cover"])}</div>'
            f'<div class="chips">{"".join(zones["chips"])}</div></div>'
        )
        return cell_html, collected

    def _render_sidebar(self, taxonomy: dict[tuple[str, str], dict[str, FieldItem]]) -> str:
        parts: list[str] = []
        for task in sorted({task for task, _ in taxonomy}):
            task_prefix = html.escape(f"{task}::", quote=True)
            parts.append(
                '<div class="node task"><div class="header"><span class="caret">▸</span>'
                f'<input type="checkbox" class="grp" data-prefix="{task_prefix}" checked>'
                f'<span class="title">{html.escape(task)}</span></div><div class="children">'
            )
            for kind in KINDS:
                items = taxonomy.get((task, kind))
                if not items:
                    continue
                kind_prefix = html.escape(f"{task}::{kind}::", quote=True)
                parts.append(
                    '<div class="node kind"><div class="header"><span class="caret">▸</span>'
                    f'<input type="checkbox" class="grp" data-prefix="{kind_prefix}" checked>'
                    f'<span class="title">{html.escape(kind)}</span></div><div class="children">'
                )
                parts.extend(self._class_row(item) for item in sorted(items.values(), key=lambda it: it.sidebar_label))
                parts.append("</div></div>")
            parts.append("</div></div>")
        return "".join(parts)

    @staticmethod
    def _class_row(item: FieldItem) -> str:
        key = html.escape(item.data_key, quote=True)
        if item.filled:
            swatch = f'<span class="swatch" style="background:{item.color}"></span>'
        else:
            swatch = f'<span class="swatch" style="background:#fff;border:2px solid {item.color}"></span>'
        return (
            f'<label class="row"><input type="checkbox" class="cls" data-key="{key}" checked>'
            f"{swatch}{html.escape(item.sidebar_label)}</label>"
        )


def _build_palettes(samples: list[SampleView]) -> dict[str, dict[str, str]]:
    """Collect each task's present class leaves (cheap pre-pass) and build its color palette."""
    present: dict[str, set[str]] = {}
    for sample in samples:
        for field_name, label in sample.fields.items():
            task, _ = _split_field(field_name)
            present.setdefault(task, set()).update(_leaves(label))
    return {task: task_palette(task, sorted(leaves)) for task, leaves in present.items()}
