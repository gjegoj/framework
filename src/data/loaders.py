"""Input loaders: one table cell into a model-ready value."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np

from src.data.registry import input_loader_registry

type InputLoader = Callable[[Any], Any]
"""Turns one table cell into a model input value (e.g. a path into an image array).

Any callable qualifies — a plain function, or one of the built-in objects below that
hold their own configuration. Nothing outside a loader needs to know whether its cells
are file paths: the loader that reads files owns its root.
"""


@input_loader_registry.register("image")
class ImageLoader:
    """Reads an image file into an ``HWC`` RGB ``uint8`` array.

    That is the raw form augmentation libraries consume; converting to a CHW
    float tensor is a transform's job, not the loader's.

    Parameters:
        root (str | Path | None): Prefix for the paths stored in the table;
            ``None`` uses them as given.
        grayscale (bool): Decode as a single ``[H, W]`` plane instead of
            ``[H, W, 3]`` RGB.
    """

    def __init__(self, root: str | Path | None = None, grayscale: bool = False) -> None:
        self._root = Path(root) if root is not None else None
        self._grayscale = grayscale

    def __call__(self, value: Any) -> np.ndarray:
        path = self._root / str(value) if self._root is not None else Path(str(value))
        flag = cv2.IMREAD_GRAYSCALE if self._grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(path), flag)
        if image is None:
            self._explain_failure(path)
        return image if self._grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _explain_failure(path: Path) -> NoReturn:
        """Turn OpenCV's silent ``None`` into a diagnosis. Runs only on failure."""
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        raise ValueError(f"Could not decode image: {path}")
