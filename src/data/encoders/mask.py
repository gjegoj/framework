"""Dense class indices: a mask file becomes an index plane before augmentation, a long tensor after."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor

from src.core import Geometry
from src.data.encoders.files import FileEncoder
from src.data.encoders.image import read_image
from src.data.encoders.label import VocabularyEncoder
from src.data.registry import target_encoder_registry


@target_encoder_registry.register("mask")
class MaskEncoder(FileEncoder, VocabularyEncoder):
    """A mask the model learns: every pixel indexes one of the task's declared classes.

    The pixel pipeline resizes the plane with its picture (nearest neighbour, never normalized) and may hand
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

    def validate(self, values: Iterable[object]) -> None:
        """Every pixel indexes a declared class; a stray index is named here, not inside a loss a thousand steps on."""
        for value in values:
            highest = int(self.load(value).max(initial=0))
            if highest >= len(self.classes):
                raise ValueError(
                    f"Mask {value!r} holds class index {highest}, but the task declares {len(self.classes)} classes "
                    f"(0..{len(self.classes) - 1}). Declare the missing classes, or remap the mask."
                )
