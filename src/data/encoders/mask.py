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
    and diagnostics as image inputs. The class count cannot be inferred without reading every
    mask, so it is declared — as ``num_classes`` or through the task's ``classes``.

    Parameters:
        num_classes (int | None): Number of classes, background included; derived from
            ``classes`` when a vocabulary is declared.
        classes (Mapping[int, str] | None): Declared vocabulary, index to name.
        root (str | Path | None): Prefix for the mask paths stored in the table.
        cache (LoaderCache | None): Serves mask reads from memory; assembly offers one.
    """

    geometry: ClassVar[Geometry] = Geometry.MASK

    def __init__(
        self,
        num_classes: int | None = None,
        classes: Mapping[int, str] | None = None,
        root: str | Path | None = None,
        cache: LoaderCache | None = None,
    ) -> None:
        names = ordered_names(classes) if classes is not None else None
        if names is None and num_classes is None:
            raise ValueError(
                "A mask needs its class count: declare 'classes' on the task, or 'num_classes' on "
                "the encoder (a dense target that is not a mask needs an explicit 'target_encoder')."
            )
        if names is not None and num_classes is not None and num_classes != len(names):
            raise ValueError(f"num_classes={num_classes} disagrees with {len(names)} declared classes; declare one.")
        resolved = num_classes if num_classes is not None else len(names or ())
        if resolved < 1:
            raise ValueError(f"num_classes must be positive, got {resolved}.")
        self._num_classes = resolved
        self._names = names
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
        names = self.class_names or [f"class{index}" for index in range(self._num_classes)]
        totals = np.zeros(self._num_classes, dtype=np.int64)
        for value in values:
            counts = np.bincount(self.load(value).reshape(-1), minlength=self._num_classes)
            if counts.size > self._num_classes:
                raise ValueError(
                    f"Mask '{value}' holds class index {counts.size - 1}, but this task declares "
                    f"{self._num_classes} classes (0..{self._num_classes - 1}). Declare the missing "
                    f"classes, or remap the mask."
                )
            totals += counts
        return ClassDistribution(counts={name: int(total) for name, total in zip(names, totals, strict=True)})

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @property
    def class_names(self) -> list[str] | None:
        return list(self._names) if self._names is not None else None
