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

    The pixel pipeline resizes the plane with its image (nearest neighbour, never normalized) and hands it
    back as an integer tensor of a width of its own choosing; ``encode`` settles the dtype a loss expects.
    """

    geometry: ClassVar[Geometry] = Geometry.MASK

    def __init__(self, *, classes: Mapping[int, str], root: str | Path | None = None) -> None:
        FileEncoder.__init__(self, root)
        VocabularyEncoder.__init__(self, classes=classes)

    def load(self, value: object) -> np.ndarray:
        """The plane as the file decoded to, which for a grayscale read is one byte a pixel.

        Widened to ``int64`` before, for a width no mask can reach: ``cv2.IMREAD_GRAYSCALE`` answers
        with eight bits whatever the file holds — measured, a 16-bit PNG comes back scaled into that
        range — so a vocabulary this encoder could ever read fits in the dtype it is already handed.
        What the eight bytes bought was arena: measured over 1000 Oxford-IIIT Pet masks, 1400.6 MiB
        held against 175.1 MiB for the same pixels, and a cache holds exactly what this returns.
        Nothing downstream wanted them either — albumentations narrows the plane to ``int32`` on its
        way through, and ``encode`` settles the ``long`` a dense loss reads.
        """
        return read_image(self.path_of(value), grayscale=True)

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
