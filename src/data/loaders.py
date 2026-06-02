"""Input loaders: convert a raw column value into a model-ready array or string.

``InputLoader`` is the port; concrete adapters implement ``load`` for each
modality and are registered in ``input_loaders`` so configs can reference them
by key.  When no loader is specified, ``_infer_loader_key`` picks one from the
actual column values — same idea as ``_infer_source_type`` for data sources:
image file extensions → "image", anything else → "text".

``file_based`` on each loader controls whether ``root_path`` is prepended to the
value before loading (True for file-path loaders, False for raw-value loaders).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.core.registry import Registry

input_loaders: Registry[InputLoader] = Registry("input_loader")

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"})


class InputLoader(ABC):
    """Converts a raw column value into a model-ready representation.

    Parameters:
        file_based (bool): When ``True`` the caller prepends ``root_path`` to
            the value before passing it to ``load``.  Set ``False`` for loaders
            that operate on raw strings rather than file paths.
    """

    file_based: bool = True

    @abstractmethod
    def load(self, value: str) -> Any:
        """Load ``value`` and return a model-ready object."""


@input_loaders.register("image")
class ImageLoader(InputLoader):
    """Loads a local image file as an ``HxWx3`` RGB uint8 numpy array."""

    file_based = True

    def load(self, path: str) -> np.ndarray:
        """Read ``path`` into an RGB uint8 array.

        Parameters:
            path (str): Filesystem path to the image.

        Returns:
            np.ndarray: RGB image array.

        Raises:
            FileNotFoundError: If the file cannot be read or decoded.
        """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@input_loaders.register("text")
class TextLoader(InputLoader):
    """Returns the raw column value as a string.

    The backbone is responsible for tokenisation; the data layer just passes
    the string through unchanged.
    """

    file_based = False

    def load(self, value: str) -> str:
        return str(value)


def _infer_loader_key(series: pd.Series) -> str:
    """Infer the ``input_loaders`` key for ``series`` from its values.

    Samples up to five non-null values and checks whether they look like image
    file paths (known extension).  Falls back to ``"text"`` for anything else.

    Parameters:
        series (pd.Series): A column from the annotation DataFrame.

    Returns:
        str: Registry key — ``"image"`` or ``"text"``.
    """
    sample = series.dropna().astype(str).head(5)
    if sample.apply(lambda v: Path(v).suffix.lower() in _IMAGE_EXTENSIONS).any():
        return "image"
    return "text"


def _normalize_inputs(
    inputs: str | dict[str, str | dict[str, str]],
) -> dict[str, str | dict[str, str]]:
    """Normalise the ``inputs`` config to a uniform dict form.

    Parameters:
        inputs: ``str`` shorthand, ``dict[alias, column]``, or
            ``dict[alias, {column, loader}]``.

    Returns:
        dict: ``{alias: column_str}`` or ``{alias: {column, loader}}``.
    """
    if isinstance(inputs, str):
        return {"image": inputs}
    return dict(inputs)
