"""Encode a boolean class mask as a base64 RGBA-PNG overlay for the ``cover`` zone.

The validated segmentation look: a class-color alpha fill inside, plus a symmetric
two-tone solid border all around — a black outer rim (contrast on any background) and a
class-color inner rim (class identity). Both gt and pred use the same solid style, so
same-class gt/pred fills stack on overlap and the intersection darkens (IoU by eye).
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

_FILL_ALPHA = 78  # ~0.31 translucent interior
_BORDER_ALPHA = 235  # near-opaque rim


def mask_overlay_uri(mask: np.ndarray, rgb: tuple[int, int, int]) -> str:
    """Encode a boolean ``[H, W]`` mask as a ``data:image/png;base64,...`` RGBA overlay."""
    inner = mask & ~_erode(mask)  # 1px class-color rim, all sides (symmetric → no offset)
    outer = _dilate(mask) & ~mask  # 1px black rim, all sides

    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = rgb
    rgba[mask, 3] = _FILL_ALPHA
    rgba[inner, 0], rgba[inner, 1], rgba[inner, 2] = rgb
    rgba[inner, 3] = _BORDER_ALPHA
    rgba[outer, 3] = _BORDER_ALPHA  # rgb stays (0, 0, 0) → black outer rim

    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _dilate(mask: np.ndarray) -> np.ndarray:
    out: np.ndarray = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
    return out


def _erode(mask: np.ndarray) -> np.ndarray:
    out: np.ndarray = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    return out
