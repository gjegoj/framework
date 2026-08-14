"""Per-task class colours: the golden angle around the hue circle."""

from __future__ import annotations

import colorsys
import zlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

GOLDEN_ANGLE: Final = 137.508
"""Successive hues land maximally far apart at any count — the reason for the angle."""

REGRESSION_COLOR: Final = "#607d8b"
"""Regression has no classes, so it takes one neutral colour outside every palette."""

FALLBACK_COLOR: Final = "#888888"
"""A leaf missing from its palette still renders — grey, and visibly unclaimed."""

_SATURATION: Final = 0.62
_LIGHTNESS: Final = 0.52


def task_palette(task: str, classes: Sequence[str]) -> dict[str, str]:
    """Map each class to a reproducible hex colour, offset per task.

    Classes are sorted, then spaced 137.508° apart around the hue circle from a
    per-task offset: deterministic, reproducible, and maximally distinct at any class
    count, which evenly-spaced slices stop being once a task has twenty classes. One
    palette colours a task's chips, masks and sidebar swatches alike, so a class is one
    colour everywhere it appears.

    The task's name seeds the starting hue, so two tasks do not mirror each other's
    palette; the sort means reordering classes in config cannot recolour a report
    somebody has already read.
    """
    offset = _hue_offset(task)
    return {name: _hsl_hex((offset + index * GOLDEN_ANGLE) % 360) for index, name in enumerate(sorted(classes))}


def _hue_offset(task: str) -> int:
    """Scatter a task name over the hue circle — a checksum, not a digest.

    ``crc32`` is the honest tool for the job: nothing here is secret, and the
    goal is only that two task names land far apart. Measured over a dozen
    plausible names, its offsets stay 5° apart at the closest, where an md5
    digest folded two of them to within 1° — the same colour to any eye.
    """
    return zlib.crc32(task.encode("utf-8")) % 360


INK_LIGHTNESS: Final = 0.30
"""The lightness a class colour is written at on a white chip.

Chosen by measurement, not by eye. The palette's own colours sit at ``0.52``,
where contrast against white runs from 1.8 to 7.4 across a twelve-class palette
— most of it below the 4.5 that 11px bold text needs, and the greens unreadable.
Re-emitting the same hue at ``0.30`` puts the whole palette between 4.8 and 13.3,
so every class passes while staying recognisably its own colour.
"""


def ink(value: str) -> str:
    """The same hue, dark enough to read on white — a class's colour as writing."""
    hue, _, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in hex_to_rgb(value)))
    return _hex(*colorsys.hls_to_rgb(hue, INK_LIGHTNESS, saturation))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` to ``(r, g, b)`` — what painting mask pixels needs."""
    digits = value.lstrip("#")
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _hsl_hex(hue_degrees: float) -> str:
    return _hex(*colorsys.hls_to_rgb(hue_degrees / 360.0, _LIGHTNESS, _SATURATION))


def _hex(red: float, green: float, blue: float) -> str:
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


DARK_INK: Final = "#10131a"
"""The dark side of ``ink_on`` — near-black with the page's own blue-grey cast."""


def ink_on(color: str) -> str:
    """White or near-black, whichever reads better on ``color`` — measured, not thresholded.

    WCAG relative-luminance contrast decides. Measured over the generated palette:
    white ink reads at 1.9–2.3:1 on the light classes (yellow ``#d0a439``, cyan
    ``#39d0d0``) where near-black reads at 8–10:1, and the ranking flips on the dark
    reds and blues (``#394cd0``: white 6.7, dark 2.8). Comparing the two measured
    contrasts has no tuning constant to drift: the palette's green sits at YIQ 148,
    one point under the luma threshold a draft used, and kept unreadable white ink.
    """
    return "#ffffff" if _contrast(color, "#ffffff") >= _contrast(color, DARK_INK) else DARK_INK


def _contrast(one: str, other: str) -> float:
    """WCAG contrast ratio between two hex colours."""
    lighter, darker = sorted((_relative_luminance(one), _relative_luminance(other)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _relative_luminance(color: str) -> float:
    def linear(channel: int) -> float:
        scaled = channel / 255
        return scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(channel) for channel in hex_to_rgb(color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue
