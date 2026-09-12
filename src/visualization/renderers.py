"""How one thing on a cell becomes markup: a class as a chip, a mask as a layer over the image.

Adding a kind of label is three edits and the type checker names two of them: the entity joins the
``Label`` union in ``entities.py``, and the two matches below stop being exhaustive until it has an
arm in each. No registry stands between them — nothing *declares* a renderer, so there is no name for
a declaration to write, and a table of four that the compiler already checks needs no mechanism.
"""

from __future__ import annotations

import html as escaping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, assert_never

from src.visualization.entities import (
    Classification,
    Classifications,
    Label,
    Regression,
    Segmentation,
    SegmentationClass,
)
from src.visualization.overlays import mask_overlay_uri
from src.visualization.palette import FALLBACK_COLOR, REGRESSION_COLOR, hex_to_rgb, ink

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.visualization.entities import Side

MAX_CHIP_CHARS = 22
"""Chip text budget before truncation; the lightbox swaps the whole text back in."""

VALUE = "value"
"""The leaf a regressed number is keyed under — it has no class to be named after."""

_MIN_RIM_ALPHA = 0.30
"""A prediction the model doubted is still a prediction, and still has an edge."""

_RIM_ALPHA_SPAN = 0.70
"""Full confidence lands on a solid rim."""

_NO_CONFIDENCE_RIM = 1.0
"""A label that expressed no confidence is not an unconfident one: a plain rim, not the faintest one."""

_SEPARATOR = "::"
"""What joins a key's parts. Named once, so the sidebar and the overlays cannot disagree."""

type Zone = Literal["cover", "chips"]
"""Which layer of a cell a piece of markup belongs to: stretched over the image, or stacked below it."""


@dataclass(frozen=True, slots=True)
class FieldContext:
    """What one label is drawn under: whose it is, which side of the cell, and in which colours."""

    task: str
    side: Side
    colors: Mapping[str, str]
    max_chip_chars: int = MAX_CHIP_CHARS
    max_side: int | None = None


@dataclass(frozen=True, slots=True)
class FieldItem:
    """One leaf of a drawn label: the overlay itself, and what the sidebar needs to switch it off."""

    task: str
    side: Side
    leaf: str
    key: str
    overlay: str
    zone: Zone
    color: str


def render_label(label: Label, context: FieldContext) -> list[FieldItem]:
    """One label as the overlays it puts on a cell; ``html.py`` never asks what kind of thing it drew."""
    match label:
        case Classification():
            return [_chip(context, label.label, _chip_text(label), label.confidence)]
        case Classifications():
            return [_chip(context, one.label, _chip_text(one), one.confidence) for one in label.classifications]
        case Regression():
            return [_chip(context, VALUE, number(label.value), color=REGRESSION_COLOR)]
        case Segmentation():
            return [_mask(context, one) for one in label.classes]
        case _:
            assert_never(label)


def leaves_of(label: Label) -> tuple[str, ...]:
    """The class-like names this label contributes — what a task's palette has to have a colour for."""
    match label:
        case Classification():
            return (label.label,)
        case Classifications():
            return tuple(one.label for one in label.classifications)
        case Regression():
            return (VALUE,)
        case Segmentation():
            return tuple(one.name for one in label.classes)
        case _:
            assert_never(label)


def field_key(task: str, side: str, leaf: str) -> str:
    """The one place the flat ``data-key`` string is joined; Python never splits it back.

    It exists for the attributes that tie a sidebar checkbox to the overlays it switches. Inside
    Python the same thing stays a structural ``(task, kind)`` pair and a leaf.
    """
    return field_prefix(task, side) + leaf


def field_prefix(task: str, side: str | None = None) -> str:
    """What every key under a task, or under one of its sides, begins with.

    The sidebar's branches select their leaves by prefix in the page's script, so the prefix has to
    be the key's own first half rather than a second string that happens to match today.
    """
    parts = (task,) if side is None else (task, side)
    return "".join(part + _SEPARATOR for part in parts)


def score_key(task: str, name: str) -> str:
    """A slider's identity: which task measured, and which way. Joined here, never split back."""
    return task + _SEPARATOR + name


def number(value: float) -> str:
    """Three decimals at most, trailing zeros dropped — the rounding chips and sliders share."""
    return f"{round(value, 3):g}"


def attr(value: str) -> str:
    """Escape for an attribute, quotes included — a class name may hold one."""
    return escaping.escape(value, quote=True)


def text(value: str) -> str:
    """Escape for element text, where a quote is a quote and needs no entity."""
    return escaping.escape(value)


def source_pill(source: str | None) -> str:
    """Where an image came from: a URL opens in a new tab, a local path copies to the clipboard."""
    if source is None:
        return ""
    escaped = attr(source)
    name = text(PurePosixPath(source).name or source)
    if source.startswith(("http://", "https://")):
        return f'<a class="src" href="{escaped}" target="_blank" rel="noopener" title="{escaped}">🔗 {name}</a>'
    return (
        f'<button class="src copy" type="button" data-copy="{escaped}" title="copy path: {escaped}">📋 {name}</button>'
    )


def _chip_text(item: Classification) -> str:
    return item.label if item.confidence is None else f"{item.label} {item.confidence:.2f}"


def _chip(
    context: FieldContext, leaf: str, caption: str, confidence: float | None = None, color: str | None = None
) -> FieldItem:
    color = color if color is not None else context.colors.get(leaf, FALLBACK_COLOR)
    key = field_key(context.task, context.side, leaf)
    overlay = (
        f'<span class="layer chip {context.side}" data-key="{attr(key)}" data-full="{attr(caption)}" '
        f'style="{_chip_style(color, context.side, confidence)}" title="{attr(caption)}">'
        f"{text(_shortened(caption, context.max_chip_chars))}</span>"
    )
    return FieldItem(context.task, context.side, leaf, key, overlay, "chips", color)


def _mask(context: FieldContext, drawn: SegmentationClass) -> FieldItem:
    color = context.colors.get(drawn.name, FALLBACK_COLOR)
    key = field_key(context.task, context.side, drawn.name)
    uri = mask_overlay_uri(drawn.mask, hex_to_rgb(color), max_side=context.max_side)
    overlay = f'<img class="layer mask" data-key="{attr(key)}" src="{uri}">'
    return FieldItem(context.task, context.side, drawn.name, key, overlay, "cover", color)


def _chip_style(color: str, side: Side, confidence: float | None) -> str:
    """Truth is filled with the class's ink; a prediction is that same ink on white, edged in its colour.

    Both sides carry one hue, so they read as one class, and what tells them apart is solid against
    outlined rather than a shade nobody can hold in mind. Confidence is the rim: faint where the model
    hesitated, solid where it did not.
    """
    written = ink(color)
    if side == "gt":
        # The ink on a filled chip is white for every class, so it is a rule of the stylesheet
        # rather than a value computed per chip; what varies is only which colour it is written on.
        return f"background:{written}"
    red, green, blue = hex_to_rgb(color)
    alpha = _NO_CONFIDENCE_RIM if confidence is None else _MIN_RIM_ALPHA + _RIM_ALPHA_SPAN * confidence
    return f"color:{written};border-color:rgba({red},{green},{blue},{alpha:.2f})"


def _shortened(value: str, budget: int) -> str:
    return value if len(value) <= budget else value[: budget - 1] + "…"
