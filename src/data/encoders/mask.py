"""Dense class indices: a mask file becomes an index plane before augmentation, a long tensor after."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from src.console import track
from src.core import ClassDistribution, Distribution, Geometry, class_name
from src.data.encoders.files import FileEncoder
from src.data.encoders.image import read_image
from src.data.encoders.label import VocabularyEncoder
from src.data.registry import target_encoder_registry


@target_encoder_registry.register("mask")
class MaskEncoder(FileEncoder, VocabularyEncoder):
    """A mask the model learns: every pixel indexes one of the task's declared classes.

    The pixel pipeline resizes the plane with its image (nearest neighbour, never normalized) and may hand
    it back as a narrower integer tensor; ``encode`` settles the dtype a loss expects.
    """

    geometry: ClassVar[Geometry] = Geometry.MASK

    def __init__(self, *, classes: Mapping[int, str], root: str | Path | None = None) -> None:
        FileEncoder.__init__(self, root)
        VocabularyEncoder.__init__(self, classes=classes)

    def load(self, value: object) -> np.ndarray:
        return read_image(self.path_of(value), grayscale=True).astype(np.int64)

    def encode(self, value: object) -> Tensor:
        return torch.as_tensor(value, dtype=torch.long)

    def distribution(self, values: Iterable[object]) -> Distribution | None:
        """Pixels per class, read from every mask — the imbalance a dense loss spends the run fighting.

        This is the one description that costs a pass over the data: measured at 1.2 ms a mask, which
        is 9 seconds for Oxford-IIIT Pet and around two minutes for a hundred thousand of them. That is
        why the summary asking for it is declared rather than given to every run.
        """
        counts = self._pixels_per_class(values, "counting mask pixels")
        return ClassDistribution(
            counts={class_name(self.classes, index): int(total) for index, total in enumerate(counts)}
        )

    def validate(self, values: Iterable[object]) -> None:
        """Every pixel indexes a declared class; a stray index is named here, not inside a loss an hour on.

        The check is the count: reading a mask is what costs, and the same read answers both questions.
        """
        self._pixels_per_class(values, "checking masks")

    def _pixels_per_class(self, values: Iterable[object], description: str) -> np.ndarray:
        """Read every mask once, refusing an index the task never declared.

        Behind a bar, because this reads the whole split: a pause with a count on it is a wait, and a
        silent one is a hang.
        """
        totals = np.zeros(len(self.classes), dtype=np.int64)
        for value in track(list(values), description=description):
            counts = np.bincount(self.load(value).reshape(-1), minlength=totals.size)
            if counts.size > totals.size:
                raise ValueError(
                    f"Mask {value!r} holds class index {counts.size - 1}, but the task declares "
                    f"{totals.size} classes (0..{totals.size - 1}). Declare the missing classes, or "
                    "remap the mask."
                )
            totals += counts
        return totals
