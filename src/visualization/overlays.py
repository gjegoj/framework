"""What a mask looks like over an image: a wash of the class's colour inside, an edge around it."""

from __future__ import annotations

from typing import Final, cast

import numpy as np

from src.visualization.png import covered_down, data_uri, shrunk_to

FILL_ALPHA: Final = 78
"""About three tenths — enough to read which class claimed the pixels, little enough to still see them."""

RIM_ALPHA: Final = 235
"""Near-opaque: the rim is the shape's edge, and an edge that washes out is not one."""

_BLACK: Final = (0, 0, 0)


def mask_overlay_uri(mask: np.ndarray, rgb: tuple[int, int, int], max_side: int | None = None) -> str:
    """A boolean ``[H, W]`` mask as a transparent PNG to lay over the image it explains.

    A translucent fill in the class's colour, a one-pixel rim of it just inside the shape, and a
    one-pixel black rim just outside — two rims because one in the class's own colour disappears over
    an image of that colour. Ground truth and prediction draw the same way, so where they overlap the
    fills darken: agreement is visible without reading a number.

    The mask is brought to display size *before* the rims are drawn: a rim drawn first survives only
    where the shrinking happened to keep it.
    """
    shown = _at_display_size(mask, max_side)
    inner_rim = shown & _neighbours_of(~shown, beyond_the_edge=True)
    outer_rim = ~shown & _neighbours_of(shown, beyond_the_edge=False)

    rgba = np.zeros((*shown.shape, 4), dtype=np.uint8)
    rgba[shown] = (*rgb, FILL_ALPHA)
    rgba[inner_rim] = (*rgb, RIM_ALPHA)
    rgba[outer_rim] = (*_BLACK, RIM_ALPHA)
    return data_uri(rgba)


def _at_display_size(mask: np.ndarray, max_side: int | None) -> np.ndarray:
    height, width = mask.shape
    target = shrunk_to(width, height, max_side)
    return mask if target is None else covered_down(mask, target)


def _neighbours_of(mask: np.ndarray, *, beyond_the_edge: bool) -> np.ndarray:
    """Pixels with a four-connected neighbour inside ``mask``, padded rather than wrapped.

    Rolling would carry the top row's neighbours onto the bottom one and draw a rim across the far
    side of the image. ``beyond_the_edge`` says what to assume outside the frame: ``False`` keeps
    the outer rim inside the image, ``True`` gives a shape running off the edge a rim along it.
    """
    padded = np.pad(mask, 1, constant_values=beyond_the_edge)
    return cast("np.ndarray", padded[2:, 1:-1] | padded[:-2, 1:-1] | padded[1:-1, 2:] | padded[1:-1, :-2])
