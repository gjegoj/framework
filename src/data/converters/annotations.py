"""Writing the ``.jsonl`` annotation format, deterministically.

One JSON object per line, one line per image, rows sorted by image path, corners rounded
to two decimals — that rounding is what makes a re-run byte-identical, so the file diffs
cleanly. The object fields are spelled from ``BoxesTargetEncoder``, so a converter cannot
write a field name the reader does not know.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.data.encoders import BoxesTargetEncoder

if TYPE_CHECKING:
    from pathlib import Path

CORNER_DECIMALS = 2
"""How precisely a corner is kept. Two decimals is a hundredth of a pixel."""

IMAGE_COLUMN = "image"
"""The row's image key, spelled once beside the object fields."""


@dataclass
class ConversionReport:
    """What a conversion did, said out loud — silent fallback is a defect.

    Counts rather than a bare success, because the interesting numbers are the ones a
    user would not have guessed: how many boxes had to be clipped back into their image,
    and which files they were in.
    """

    images: int = 0
    objects: int = 0
    clipped: int = 0
    dropped_crowd: int = 0
    clipped_files: set[str] = field(default_factory=set)
    """Files, not boxes — a file with nine clipped boxes is one place to look; the
    per-box count is ``clipped``."""

    def summary(self) -> str:
        """One line for a terminal: what was written, and what had to be corrected."""
        parts = [f"{self.images} images, {self.objects} objects"]
        if self.clipped:
            where = ", ".join(sorted(self.clipped_files))
            parts.append(f"clipped {self.clipped} out-of-bounds box(es) in {where}")
        if self.dropped_crowd:
            parts.append(f"dropped {self.dropped_crowd} crowd object(s)")
        return "; ".join(parts) + "."


def annotation_object(corners: tuple[float, float, float, float], name: str) -> dict[str, Any]:
    """One object in the canonical shape: its box in pixels, its class by name."""
    return {
        BoxesTargetEncoder.BOX: [round(corner, CORNER_DECIMALS) for corner in corners],
        BoxesTargetEncoder.CLASS: name,
    }


def annotation_row(image: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
    """One annotation row: which image, and the objects kept for it."""
    return {IMAGE_COLUMN: image, "objects": objects}


def clipped_box(
    corners: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    image: str,
    report: ConversionReport,
) -> tuple[float, float, float, float]:
    """The box bounded by its image, counting the correction; refused when nothing is left.

    Clipping is the right repair for the usual cause — an annotation tool that let a
    drag run past the edge — but a box that is empty *after* clipping was never a box,
    and training on it would teach a degenerate target. That one refuses by image path,
    which is the only thing that lets a human find it.
    """
    x1, y1, x2, y2 = corners
    bounded = (max(x1, 0.0), max(y1, 0.0), min(x2, float(width)), min(y2, float(height)))
    if bounded != corners:
        report.clipped += 1
        report.clipped_files.add(image)
    if bounded[2] - bounded[0] <= 0 or bounded[3] - bounded[1] <= 0:
        raise ValueError(
            f"Box {[round(corner, CORNER_DECIMALS) for corner in corners]} in '{image}' has no area "
            f"inside the {width}x{height} image. Fix the annotation."
        )
    return bounded


def write_annotations(records: list[dict[str, Any]], into: Path) -> None:
    """The rows as JSON Lines, sorted by image path so the file is reproducible."""
    into.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: str(record[IMAGE_COLUMN]))
    lines = [json.dumps(record, ensure_ascii=False) for record in ordered]
    into.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
