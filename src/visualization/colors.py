"""Per-task color mapping for the visualization renderer.

Replaces a single global palette: within a task, classes are sorted and spaced around
the hue circle by the golden angle (maximally distinct for any class count — important
for segmentation), from a per-task hue offset so each task has its own gamut. Colors are
deterministic and reproducible. One palette colors a task's chips, masks, and swatches.
"""

from __future__ import annotations

import colorsys
import hashlib

GOLDEN_ANGLE = 137.508
REGRESSION_COLOR = "#607d8b"  # neutral, not palette-driven (regression has no classes)
FALLBACK_COLOR = "#888888"  # used when a leaf is missing from the task palette
_SATURATION = 0.62
_LIGHTNESS = 0.52


def task_palette(task: str, classes: list[str]) -> dict[str, str]:
    """Map each class to a maximally-distinct, reproducible hex color for one ``task``.

    Parameters:
        task (str): Task name; its hash seeds a per-task hue offset.
        classes (list[str]): Class names (sorted internally, so input order is irrelevant).

    Returns:
        dict[str, str]: ``class name -> "#rrggbb"``.
    """
    offset = int(hashlib.md5(task.encode("utf-8")).hexdigest(), 16) % 360
    return {
        cls: _hsl_hex((offset + i * GOLDEN_ANGLE) % 360, _SATURATION, _LIGHTNESS)
        for i, cls in enumerate(sorted(classes))
    }


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert ``"#rrggbb"`` to an ``(r, g, b)`` int triple (for mask pixel coloring)."""
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _hsl_hex(hue_deg: float, saturation: float, lightness: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue_deg / 360.0, lightness, saturation)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
