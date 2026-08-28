"""Per-pixel targets read from mask files."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, override

import numpy as np

from src.core.entities import ClassDistribution, Distribution
from src.core.taxonomy import Geometry
from src.core.vocabulary import ordered_names
from src.data.cache import cached
from src.data.encoders.base import TargetEncoder
from src.data.loaders import ImageLoader
from src.data.registry import target_encoder_registry

if TYPE_CHECKING:
    from src.data.cache import LoaderCache
    from src.data.loaders import InputLoader

log = logging.getLogger(__name__)


@target_encoder_registry.register("mask")
class MaskTargetEncoder(TargetEncoder):
    """Segmentation masks: an image file of class indices into an ``[H, W]`` array.

    Reading is delegated to a grayscale ``ImageLoader``, so masks get the same root handling
    and diagnostics as image inputs. The vocabulary sizes the index map: background
    included, as the mask files number it.

    Parameters:
        classes (Mapping[int, str]): The vocabulary, index to name.
        root (str | Path | None): Prefix for the mask paths stored in the table.
        cache (LoaderCache | None): Serves mask reads from memory; assembly offers one.
    """

    geometry: ClassVar[Geometry] = Geometry.MASK

    def __init__(
        self,
        classes: Mapping[int, str],
        root: str | Path | None = None,
        cache: LoaderCache | None = None,
    ) -> None:
        self._names = ordered_names(classes)
        # The mask is read through a loader this encoder owns, so caching has to be
        # handed in: there is nothing on the outside left to wrap.
        read: InputLoader = ImageLoader(root=root, grayscale=True)
        self._read = cached(read, cache) if cache is not None else read

    @override
    def load(self, value: Any) -> np.ndarray:
        """The mask file as an ``[H, W]`` index map — pixels, so geometry can move them."""
        mask: np.ndarray = self._read(value).astype(np.int64)
        return mask

    def encode(self, value: Any) -> np.ndarray:
        """Already its training form: ``load`` did the reading, the transform the geometry."""
        return np.asarray(value)

    @override
    def distribution(self, values: Iterable[Any]) -> Distribution | None:
        """Pixels per class, read from every mask — the class imbalance a loss will fight.

        Counted in full. Measured: 0.88 ms per mask, so 3.3 s for 3680 masks, once; with a cache
        the reads are the ones training is about to warm. A pixel outside the declared vocabulary
        is refused here rather than as a shape error at the loss. Reads through ``load``, which
        is what reading a cell is.
        """
        totals = np.zeros(len(self._names), dtype=np.int64)
        for value in values:
            counts = np.bincount(self.load(value).reshape(-1), minlength=len(self._names))
            if counts.size > len(self._names):
                raise ValueError(
                    f"Mask '{value}' holds class index {counts.size - 1}, but this task declares "
                    f"{len(self._names)} classes (0..{len(self._names) - 1}). Declare the missing "
                    f"classes, or remap the mask."
                )
            totals += counts
        return ClassDistribution(counts={name: int(total) for name, total in zip(self._names, totals, strict=True)})

    @property
    def num_classes(self) -> int:
        return len(self._names)

    @property
    def class_names(self) -> list[str]:
        return list(self._names)
