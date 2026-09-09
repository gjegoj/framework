"""Reading a drawn page back: the picture it embeds, and a batch whose picture is known in advance.

A samples page is the only public face of the grid's de-normalisation, so a test about
the numbers it undoes reads the pixels off the page rather than the attribute they came from.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence

import cv2
import numpy as np
import torch

from src.core import Batch

COLOUR: tuple[int, int, int] = (51, 128, 204)
"""The RGB every pixel of ``colour_batch`` shows once de-normalised: (0.2, 0.5, 0.8) of 255."""

_EMBEDDED_PNG = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")


def first_image(page: str) -> np.ndarray:
    """The first picture on the page, decoded to an RGB array."""
    embedded = _EMBEDDED_PNG.search(page)
    assert embedded is not None, "the page embeds no picture"
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(embedded.group(1)), np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None, "the embedded picture is not a PNG cv2 can read"
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def colour_batch(mean: Sequence[float], std: Sequence[float], rows: int = 4) -> Batch:
    """Four 2x2 images of one colour, normalised by ``mean``/``std`` the way the transforms would.

    Drawn by a grid that undoes the same numbers, every pixel comes back as ``COLOUR``;
    drawn by one that undoes different numbers, none does.
    """
    value = torch.tensor([channel / 255 for channel in COLOUR]).view(1, 3, 1, 1)
    normalised = (value - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    return Batch(
        inputs={"image": normalised.expand(rows, 3, 2, 2).clone()},
        targets={"label": torch.tensor([index % 2 for index in range(rows)])},
        meta={"cells": [{"image": f"images/{index}.png"} for index in range(rows)]},
    )
