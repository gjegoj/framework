"""Pixels into a page: a PNG data URI, and the size one is shown at.

Written here rather than taken from an imaging library because this package is meant to be lifted out
whole, and a page needs exactly two things of a codec — write these bytes, and fit them to a box. PNG
is a container around zlib, so both are a few dozen lines of numpy and the standard library, and the
library's only dependency stays numpy.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import cast

import numpy as np

PREFIX = "data:image/png;base64,"
"""What a browser needs in front of the bytes to show them without fetching anything."""

_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_COLOR_TYPES = {3: 2, 4: 6}
"""Channels → PNG's own name for what they mean: three are colour, four are colour with transparency."""

_BIT_DEPTH = 8

_UP_FILTER = 2
"""The per-row filter byte: every row is stored as its difference from the row above.

One filter for every row, rather than the per-row heuristic an imaging library runs. Measured at
224x224 against Pillow, which picks adaptively: a photograph reaches 114.7 KB here against its
113.3 KB, and a mask overlay 0.7 KB against 0.6 KB. Storing the rows unfiltered instead costs 140.5 KB
on the same photograph — a fifth of the page's weight for the two lines this takes.
"""


def data_uri(pixels: np.ndarray, max_side: int | None = None) -> str:
    """Encode a uint8 ``[H, W, 3]`` or ``[H, W, 4]`` array as ``data:image/png;base64,...``.

    ``max_side`` bounds what goes into the page rather than what the tensor holds: a grid inlines a
    picture and every mask over it for each cell, so the page's weight is the cell count times the
    layer count times this. Shrinking averages, because a downscaled photograph that sampled one pixel
    per block would report detail it does not have.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] not in _COLOR_TYPES:
        raise ValueError(
            f"A page draws uint8 [H, W, 3] or [H, W, 4] pixels; got {pixels.dtype} of shape {pixels.shape}."
        )
    target = shrunk_to(pixels.shape[1], pixels.shape[0], max_side)
    return PREFIX + base64.b64encode(_png(pixels if target is None else _averaged_down(pixels, target))).decode("ascii")


def shrunk_to(width: int, height: int, max_side: int | None) -> tuple[int, int] | None:
    """The size something is shown at, or ``None`` when it already fits.

    Answered for the picture and for every mask over it, so both land on the same grid of pixels and
    an overlay cannot drift a row off the shape it is explaining.
    """
    if max_side is None or max(width, height) <= max_side:
        return None
    scale = max_side / max(width, height)
    # A side that rounded to nothing is a picture no decoder will take, and the page would carry the
    # error instead of the sample.
    return max(1, round(width * scale)), max(1, round(height * scale))


def covered_down(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Shrink a boolean mask so that a block the class covered any of keeps it.

    A mask says which pixels a class claims, so it cannot be averaged — that would invent pixels the
    class only partly holds. Taking one pixel per block instead loses thin shapes outright: measured
    at the shipped bound, half of the one-pixel-wide columns of a 512-pixel mask disappear, while the
    class keeps its row in the sidebar and its colour in the legend. The page then says a class is
    there and draws nothing. Nothing is invented by this either — a block that lights up held it.
    """
    width, height = size
    return _covered(_covered(mask, axis=0, target=height), axis=1, target=width)


def _covered(mask: np.ndarray, axis: int, target: int) -> np.ndarray:
    """One axis reduced to ``target`` entries, each true where its block held anything."""
    length = mask.shape[axis]
    if target >= length:
        return mask
    starts = (np.arange(target) * length) // target
    return np.logical_or.reduceat(mask, starts, axis=axis)


def _averaged_down(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Every target pixel is the mean of the block it covers — box sampling, in two passes."""
    width, height = size
    averaged = _averaged(_averaged(pixels.astype(np.float32), axis=0, target=height), axis=1, target=width)
    return np.clip(np.round(averaged), 0, 255).astype(np.uint8)


def _averaged(values: np.ndarray, axis: int, target: int) -> np.ndarray:
    """One axis reduced to ``target`` entries, each the mean of its block.

    ``reduceat`` sums between the given starts, which is the block boundary itself; the blocks differ
    in length wherever the ratio is not whole, so each sum is divided by its own.
    """
    length = values.shape[axis]
    if target >= length:
        return values
    starts = (np.arange(target) * length) // target
    counts = np.diff(np.append(starts, length)).astype(np.float32)
    spread = [1] * values.ndim
    spread[axis] = target
    return np.asarray(np.add.reduceat(values, starts, axis=axis) / counts.reshape(spread))


def _png(pixels: np.ndarray) -> bytes:
    """The three chunks a viewer needs: what this is, the rows themselves, and the end."""
    height, width, channels = pixels.shape
    header = struct.pack(">IIBBBBB", width, height, _BIT_DEPTH, _COLOR_TYPES[channels], 0, 0, 0)
    rows = np.hstack([np.full((height, 1), _UP_FILTER, dtype=np.uint8), _up(pixels.reshape(height, -1))])
    return b"".join(
        (
            _SIGNATURE,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(rows.tobytes())),
            _chunk(b"IEND", b""),
        )
    )


def _up(rows: np.ndarray) -> np.ndarray:
    """Every row minus the one above it, wrapping at 256 as the format defines the subtraction.

    A first row of zeros rather than a special case: subtracting nothing leaves it as it is.
    """
    above = np.vstack([np.zeros((1, rows.shape[1]), dtype=np.uint8), rows[:-1]])
    return cast("np.ndarray", ((rows.astype(np.int16) - above) % 256).astype(np.uint8))


def _chunk(tag: bytes, payload: bytes) -> bytes:
    """Length, name, bytes, and the checksum over the last two — PNG's own frame around everything."""
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))
