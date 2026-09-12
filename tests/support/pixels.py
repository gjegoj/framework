"""Reading a page's pixels back, with a decoder that is not the one under test."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from src.visualization.png import PREFIX


def decoded(uri: str) -> np.ndarray:
    """The image a browser would show, as RGB or RGBA — cv2 reads BGR, so the channels come back."""
    assert uri.startswith(PREFIX)
    raw = np.frombuffer(base64.b64decode(uri.removeprefix(PREFIX)), dtype=np.uint8)
    read = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    assert read is not None, "the bytes are not an image any decoder will take"
    return np.asarray(read[..., ::-1] if read.shape[2] == 3 else read[..., [2, 1, 0, 3]])
