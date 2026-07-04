"""Input loaders: convert a raw column value into a model-ready array or string.

``InputLoader`` is the port; concrete adapters implement ``load`` for each
modality and are registered in ``input_loaders`` so configs can reference them
by key.  When no loader is specified, ``infer_loader_key`` picks one from the
actual column values — the same extension-based inference the wiring layer uses
to choose a data-source format: image file extensions → "image", anything else → "text".

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

from src.data.registry import input_loaders

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"})
_EMBEDDING_EXTENSIONS = frozenset({".npy"})


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


@input_loaders.register("embedding")
class EmbeddingLoader(InputLoader):
    """Loads a precomputed embedding vector from a ``.npy`` file as ``[D]`` float32.

    One vector per file — the simplest precomputed-embedding layout. A future
    "matrix + index" loader (all vectors in a single ``[N, D]`` file, addressed by
    row index, ``file_based=False``) would be a separate registered loader holding a
    shared handle to the matrix; it is intentionally not built here.
    """

    file_based = True

    def load(self, path: str) -> np.ndarray:
        """Read ``path`` into a 1-D float32 array.

        Parameters:
            path (str): Filesystem path to a ``.npy`` file holding one vector.

        Returns:
            np.ndarray: The embedding vector ``[D]`` as float32.

        Raises:
            ValueError: If the file does not contain a 1-D array.
        """
        vector: np.ndarray = np.load(path)
        if vector.ndim != 1:
            raise ValueError(f"EmbeddingLoader expects a 1-D vector, got shape {vector.shape} from {path}.")
        return vector.astype(np.float32)


def infer_loader_key(series: pd.Series) -> str:
    """Infer the ``input_loaders`` key for ``series`` from its values.

    Samples up to five non-null values and inspects their file extension: image
    extensions → ``"image"``, ``.npy`` → ``"embedding"``.  Falls back to ``"text"``
    for anything else.

    Parameters:
        series (pd.Series): A column from the annotation DataFrame.

    Returns:
        str: Registry key — ``"image"``, ``"embedding"`` or ``"text"``.
    """
    suffixes = series.dropna().astype(str).head(5).apply(lambda v: Path(v).suffix.lower())
    if suffixes.isin(_IMAGE_EXTENSIONS).any():
        return "image"
    if suffixes.isin(_EMBEDDING_EXTENSIONS).any():
        return "embedding"
    return "text"


def normalize_inputs(
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


def input_aliases(inputs: str | dict[str, str | dict[str, str]]) -> tuple[str, ...]:
    """Ordered input alias names from the ``data.inputs`` config.

    ``str`` shorthand → the single ``image`` alias; a dict → its keys in declaration
    order. The single source of truth shared by task wiring (MULTIVIEW/MULTISTREAM key
    derivation) and export planning (the traced input alias).
    """
    return tuple(normalize_inputs(inputs).keys())
