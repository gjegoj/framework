"""Everything the page shows as pixels: the sample's own image, and masks over it."""

from __future__ import annotations

import base64
import io
from typing import cast

import numpy as np
from PIL import Image
from PIL.Image import Resampling

_FILL_ALPHA = 78
"""~0.31 — enough to read the class, little enough to keep the image visible."""

_RIM_ALPHA = 235
"""Near-opaque: the rim is the shape's edge and must not wash out."""

_BLACK = (0, 0, 0)


def png_data_uri(pixels: np.ndarray, max_side: int | None = None) -> str:
    """Encode a uint8 ``[H, W, 3]`` or ``[H, W, 4]`` array as ``data:image/png;base64,...``.

    ``max_side`` bounds what goes into the page: measured, eight cells of a ten-class
    segmentation at full 512px inline to a 49 MB page in 13 seconds. Downscaling is smooth
    for pictures and nearest for masks, because a mask that interpolates stops being a mask.
    """
    image = Image.fromarray(pixels)
    target = _fitted(image.width, image.height, max_side)
    if target is not None:
        smooth = pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3
        image = image.resize(target, Resampling.LANCZOS if smooth else Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def mask_overlay_uri(mask: np.ndarray, rgb: tuple[int, int, int], max_side: int | None = None) -> str:
    """Encode a boolean ``[H, W]`` mask as a ``data:image/png;base64,...`` overlay.

    A translucent class-colour fill, a 1px class-colour inner rim, a 1px black outer rim so
    the shape holds against any image; ground truth and prediction share the style, so
    overlapping fills darken — IoU by eye. The mask is resized to display size *before* the
    rims are drawn: measured, shrinking a rimmed 512 mask to 256 erased two of four sides,
    because nearest-neighbour sampling drops whole rows.
    """
    shown = _at_display_size(mask, max_side)
    inner_rim = shown & _neighbours_of(~shown, beyond_the_edge=True)
    outer_rim = ~shown & _neighbours_of(shown, beyond_the_edge=False)

    rgba = np.zeros((*shown.shape, 4), dtype=np.uint8)
    rgba[shown] = (*rgb, _FILL_ALPHA)
    rgba[inner_rim] = (*rgb, _RIM_ALPHA)
    rgba[outer_rim] = (*_BLACK, _RIM_ALPHA)
    return png_data_uri(rgba)


def _at_display_size(mask: np.ndarray, max_side: int | None) -> np.ndarray:
    """The mask at the size the page shows it, sampled nearest so it stays a mask."""
    height, width = mask.shape
    target = _fitted(width, height, max_side)
    if target is None:
        return mask
    shrunk = Image.fromarray(mask.astype(np.uint8)).resize(target, Resampling.NEAREST)
    return np.asarray(shrunk) > 0


def _fitted(width: int, height: int, max_side: int | None) -> tuple[int, int] | None:
    """The size something is shown at, or ``None`` when it already fits."""
    if max_side is None or max(width, height) <= max_side:
        return None
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _neighbours_of(mask: np.ndarray, *, beyond_the_edge: bool) -> np.ndarray:
    """Pixels with a 4-connected neighbour inside ``mask``, padded rather than wrapped.

    ``np.roll`` would carry the top row's neighbours onto the bottom one, drawing
    a rim across the far side of the image. ``beyond_the_edge`` says what to
    assume outside the frame: ``False`` keeps the outer rim inside the image,
    ``True`` gives a mask running off the edge an inner rim along it.
    """
    padded = np.pad(mask, 1, constant_values=beyond_the_edge)
    return cast("np.ndarray", padded[2:, 1:-1] | padded[:-2, 1:-1] | padded[1:-1, 2:] | padded[1:-1, :-2])
