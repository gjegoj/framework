"""Per-class colours: one for telling classes apart, one for writing a class's name.

Two readings of one hue, because the two jobs pull opposite ways. A mask over a photograph and a
swatch in a sidebar want colours that separate at a glance, which is a mid-lightness palette. Text at
chip size wants contrast against its ground, which the same palette does not have. So the hue is the
class's identity and the lightness is chosen per job.
"""

from __future__ import annotations

import colorsys
import zlib
from collections.abc import Sequence
from typing import Final

GOLDEN_ANGLE: Final = 137.508
"""Successive hues land maximally far apart at any count — the reason for this angle and no other."""

REGRESSION_COLOR: Final = "#607d8b"
"""A regressed number has no classes, so it takes one neutral colour outside every palette."""

FALLBACK_COLOR: Final = "#888888"
"""A leaf missing from its palette still draws — grey, and visibly unclaimed."""

_SATURATION: Final = 0.62
_LIGHTNESS: Final = 0.52
"""What separates twelve classes from each other, which is a different job from being readable."""

INK_LIGHTNESS: Final = 0.28
"""The lightness a class's hue is re-emitted at to be read as text, either way round.

Contrast is symmetric, so one number serves both chips: the class written on white, and white written
on the class. Chosen by sweeping the whole hue circle rather than the classes of one page — at 0.30
the worst hue (a yellow-green) reaches only 4.4:1 against white, under the 4.5 that 11px text needs,
and at 0.32 only 4.0:1. At this value the worst hue reaches 5.0:1. Held by the test, not by this note.
"""


def task_palette(task: str, classes: Sequence[str]) -> dict[str, str]:
    """Map each class to a reproducible colour, offset per task.

    Classes are sorted, then spaced ``GOLDEN_ANGLE`` apart around the hue circle from an offset
    seeded by the task's name: deterministic, distinct at any class count, and one colour per class
    everywhere it appears. Reordering classes in a config cannot recolour a report somebody has read.
    """
    offset = _hue_offset(task)
    return {name: _hsl_hex((offset + index * GOLDEN_ANGLE) % 360) for index, name in enumerate(sorted(classes))}


def ink(value: str) -> str:
    """The same hue, dark enough to be read against white and to have white read on it."""
    hue, _, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in hex_to_rgb(value)))
    return _hex(*colorsys.hls_to_rgb(hue, INK_LIGHTNESS, saturation))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` as the three numbers a mask is painted with."""
    digits = value.lstrip("#")
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _hue_offset(task: str) -> int:
    """Scatter a task's name over the hue circle — a checksum, not a digest.

    Nothing here is secret; the only requirement is that two task names land far apart, which is
    what a checksum over a short string already does.
    """
    return zlib.crc32(task.encode("utf-8")) % 360


def _hsl_hex(hue_degrees: float) -> str:
    return _hex(*colorsys.hls_to_rgb(hue_degrees / 360.0, _LIGHTNESS, _SATURATION))


def _hex(red: float, green: float, blue: float) -> str:
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
