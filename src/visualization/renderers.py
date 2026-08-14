"""How one thing on a cell becomes HTML: one kind of label, one named renderer.

To add a kind (a heatmap, detection boxes), every step is a named place:

1. the entity joins the ``Label`` union in ``entities.py``;
2. a ``LabelRenderer`` subclass here, registered under the entity's type;
3. an annotation objective/topology in ``annotators.py`` produces it;
4. ``test_renderers.py``'s exhaustiveness pin goes green again.
"""

from __future__ import annotations

import html as html_escape
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, override

from src.visualization.entities import (
    Classification,
    Classifications,
    Image,
    Label,
    Media,
    Regression,
    Segmentation,
    Text,
)
from src.visualization.overlays import mask_overlay_uri, png_data_uri
from src.visualization.palette import FALLBACK_COLOR, REGRESSION_COLOR, hex_to_rgb, ink, ink_on
from src.visualization.registry import label_renderer_registry, media_renderer_registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.visualization.entities import Kind

MAX_CHIP_CHARS = 22
"""Chip text budget before truncation; the lightbox swaps the full text back in."""

_MIN_RIM_ALPHA = 0.30
"""A prediction the model doubted is still a prediction, and still has an edge."""

_RIM_ALPHA_SPAN = 0.70
"""Full confidence lands on a solid rim."""

_NO_CONFIDENCE_RIM = 1.0
"""A label that expressed no confidence is not an unconfident one.

A regression chip has no confidence to encode, so taking the bottom of the ramp
for it said the opposite of the truth. It gets a plain solid rim: no reading, not
the lowest reading.
"""

type Zone = Literal["frame", "cover", "chips", "caption"]
"""Which of a cell's four layers a piece of markup belongs to.

Media and labels both name a zone rather than being sorted by type, so a cell
assembles the same way whatever it holds: pictures fill the frame, masks stretch
over it, chips and captions stack at the foot.
"""

ZONES: tuple[Zone, ...] = ("frame", "cover", "chips", "caption")


_SEPARATOR = "::"
"""What joins a key's parts. Named once so the sidebar and the overlays cannot disagree."""


def field_key(task: str, kind: str, leaf: str) -> str:
    """The one place the flat ``data-key`` string is glued; Python never re-splits it.

    ``task::kind::leaf`` exists for the ``data-key`` attributes that tie a sidebar
    checkbox to its overlays. Inside Python the key stays a structural tuple, so nothing
    has to parse back what something else glued.
    """
    return field_prefix(task, kind) + leaf


def field_prefix(task: str, kind: str | None = None) -> str:
    """What every key under a task, or under one of its sides, starts with.

    The sidebar's branches select their leaves by prefix in the page's script, so
    the prefix has to be the key's own first half rather than a second f-string
    that happens to match today.
    """
    parts = (task,) if kind is None else (task, kind)
    return "".join(part + _SEPARATOR for part in parts)


def score_key(task: str, name: str) -> str:
    """A slider's identity: which task measured, and which way. Glued here, never re-split."""
    return task + _SEPARATOR + name


@dataclass(frozen=True, slots=True)
class FieldContext:
    """What one label is rendered under: whose it is, which side, which colours."""

    task: str
    kind: Kind
    colors: Mapping[str, str]
    max_chip_chars: int = MAX_CHIP_CHARS
    max_side: int | None = None


@dataclass(frozen=True, slots=True)
class MediaItem:
    """One drawn input, the zone it belongs to, and the shape it wants to be drawn at.

    ``aspect`` is width over height, and ``None`` for a medium with no geometry of
    its own. The cell takes the first one it is offered, so a new medium that has
    pixels declares its shape the same way it declares its zone — additively, with
    the renderer never asking what kind of thing it is.
    """

    markup: str
    zone: Zone
    aspect: float | None = None


@dataclass(frozen=True, slots=True)
class FieldItem:
    """One leaf of a rendered label: a toggleable overlay and the sidebar row that toggles it."""

    task: str
    kind: Kind
    leaf: str
    key: str
    overlay: str
    zone: Zone
    color: str


class LabelRenderer[L: Label](ABC):
    """Everything one kind of label knows about becoming HTML.

    ``leaves`` runs first, across every sample, so the palette can colour a
    class the same on every cell; ``render`` runs per cell with those colours.
    Both live on one class so a kind cannot register one and forget the other.
    """

    @abstractmethod
    def leaves(self, label: L) -> tuple[str, ...]:
        """The class-like names this label contributes — what a task's palette must colour."""

    @abstractmethod
    def render(self, label: L, context: FieldContext) -> list[FieldItem]:
        """This label as overlays; ``html.py`` never asks what kind of thing it draws."""


class MediaRenderer[M: Media](ABC):
    """One kind of input, drawn — a modality is not a config choice."""

    @abstractmethod
    def render(self, media: M, alias: str, max_side: int | None) -> MediaItem: ...


@label_renderer_registry.register(Classification)
class ClassificationRenderer(LabelRenderer[Classification]):
    @override
    def leaves(self, label: Classification) -> tuple[str, ...]:
        return (label.label,)

    @override
    def render(self, label: Classification, context: FieldContext) -> list[FieldItem]:
        color = context.colors.get(label.label, FALLBACK_COLOR)
        return [_chip(context, label.label, _said(label), color, label.confidence)]


@label_renderer_registry.register(Classifications)
class ClassificationsRenderer(LabelRenderer[Classifications]):
    @override
    def leaves(self, label: Classifications) -> tuple[str, ...]:
        return tuple(item.label for item in label.classifications)

    @override
    def render(self, label: Classifications, context: FieldContext) -> list[FieldItem]:
        return [
            _chip(context, item.label, _said(item), context.colors.get(item.label, FALLBACK_COLOR), item.confidence)
            for item in label.classifications
        ]


@label_renderer_registry.register(Regression)
class RegressionRenderer(LabelRenderer[Regression]):
    @override
    def leaves(self, label: Regression) -> tuple[str, ...]:
        return ("value",)

    @override
    def render(self, label: Regression, context: FieldContext) -> list[FieldItem]:
        """Just the number. How far the prediction missed is the verdict's score, printed once."""
        return [_chip(context, "value", number(label.value), REGRESSION_COLOR)]


@label_renderer_registry.register(Segmentation)
class SegmentationRenderer(LabelRenderer[Segmentation]):
    @override
    def leaves(self, label: Segmentation) -> tuple[str, ...]:
        return tuple(entry.name for entry in label.classes)

    @override
    def render(self, label: Segmentation, context: FieldContext) -> list[FieldItem]:
        items: list[FieldItem] = []
        for entry in label.classes:
            color = context.colors.get(entry.name, FALLBACK_COLOR)
            key = field_key(context.task, context.kind, entry.name)
            uri = mask_overlay_uri(entry.mask, hex_to_rgb(color), max_side=context.max_side)
            overlay = f'<img class="layer mask" data-key="{attr(key)}" src="{uri}">'
            items.append(_item(context, entry.name, key, overlay, "cover", color))
        return items


@media_renderer_registry.register(Image)
class ImageRenderer(MediaRenderer[Image]):
    @override
    def render(self, media: Image, alias: str, max_side: int | None) -> MediaItem:
        uri = png_data_uri(media.pixels, max_side=max_side)
        picture = f'<img class="picture" alt="{attr(alias)}" src="{uri}">'
        height, width = media.pixels.shape[:2]
        return MediaItem(markup=picture + _source_pill(media.source), zone="frame", aspect=width / height)


@media_renderer_registry.register(Text)
class TextRenderer(MediaRenderer[Text]):
    @override
    def render(self, media: Text, alias: str, max_side: int | None) -> MediaItem:
        """A strip at the foot of the cell; the alias names which input it is."""
        strip = f'<div class="caption"><span class="alias">{text(alias)}</span>{text(media.text)}</div>'
        return MediaItem(markup=strip, zone="caption")


def render_label(label: Label, context: FieldContext) -> list[FieldItem]:
    """One label → its overlays. The type chooses the renderer: config never does.

    An unregistered kind fails inside the registry, which names what *is*
    registered — no hand-written ``known:`` list to fall out of date.
    """
    # `key: type` because mypy will not match a union of `type[...]`s against
    # the Hashable protocol on its own; the widening is the whole fix.
    key: type = type(label)
    renderer: LabelRenderer[Any] = label_renderer_registry.create(key)
    return renderer.render(label, context)


def leaves_of(label: Label) -> tuple[str, ...]:
    """The class-like names a label contributes — what a task's palette must colour."""
    key: type = type(label)
    renderer: LabelRenderer[Any] = label_renderer_registry.create(key)
    return renderer.leaves(label)


def render_media(media: Media, alias: str, max_side: int | None = None) -> MediaItem:
    """One input, drawn — same dispatch, smaller family."""
    key: type = type(media)
    renderer: MediaRenderer[Any] = media_renderer_registry.create(key)
    return renderer.render(media, alias, max_side)


def number(value: float) -> str:
    """Three decimals at most, trailing zeros dropped — the same rounding the sliders use."""
    return f"{round(value, 3):g}"


def _said(item: Classification) -> str:
    return item.label if item.confidence is None else f"{item.label} {item.confidence:.2f}"


def _chip(
    context: FieldContext,
    leaf: str,
    said: str,
    color: str,
    confidence: float | None = None,
) -> FieldItem:
    key = field_key(context.task, context.kind, leaf)
    overlay = (
        f'<span class="layer chip {context.kind}" data-key="{attr(key)}" data-full="{attr(said)}" '
        f'style="{_chip_style(color, context.kind, confidence)}" title="{attr(said)}">'
        f"{text(_shorten(said, context.max_chip_chars))}</span>"
    )
    return _item(context, leaf, key, overlay, "chips", color)


def _item(context: FieldContext, leaf: str, key: str, overlay: str, zone: Zone, color: str) -> FieldItem:
    return FieldItem(
        task=context.task,
        kind=context.kind,
        leaf=leaf,
        key=key,
        overlay=overlay,
        zone=zone,
        color=color,
    )


def _chip_style(color: str, kind: Kind, confidence: float | None) -> str:
    """Ground truth is the class colour; a prediction is that colour written on white.

    The two used to differ by fill alone — a solid chip against a translucent one
    of the same colour — and a translucent chip over an arbitrary image is hard to
    read whatever is behind it. Now the backing is always white, the class colour
    is the ink, and the two sides stay tied by hue.

    Confidence moved to the rim, which frees the backing to stay opaque: a hesitant
    prediction has a faint edge and a confident one a solid edge, and the number is
    legible either way.

    Ground truth's ink is chosen by measured contrast, not fixed white: a fixed
    white read at 1.9:1 on the palette's light classes — see ``ink_on``.
    """
    if kind == "gt":
        return f"background:{color};color:{ink_on(color)}"
    red, green, blue = hex_to_rgb(color)
    alpha = _NO_CONFIDENCE_RIM if confidence is None else _MIN_RIM_ALPHA + _RIM_ALPHA_SPAN * confidence
    return f"color:{ink(color)};border-color:rgba({red},{green},{blue},{alpha:.2f})"


def _shorten(value: str, budget: int) -> str:
    return value if len(value) <= budget else value[: budget - 1] + "…"


def attr(value: str) -> str:
    """Escape a value for an HTML attribute, quotes included — a class name may hold one."""
    return html_escape.escape(value, quote=True)


def text(value: str) -> str:
    """Escape a value for element text, where a quote is a quote and needs no entity."""
    return html_escape.escape(value)


def _source_pill(source: str | None) -> str:
    """A URL opens in a new tab; a local path copies to the clipboard instead."""
    if source is None:
        return ""
    escaped = attr(source)
    name = text(PurePosixPath(source).name or source)
    if source.startswith(("http://", "https://")):
        return f'<a class="src" href="{escaped}" target="_blank" rel="noopener" title="{escaped}">🔗 {name}</a>'
    return (
        f'<button class="src copy" type="button" data-copy="{escaped}" title="copy path: {escaped}">📋 {name}</button>'
    )
