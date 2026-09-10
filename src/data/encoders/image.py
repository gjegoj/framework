"""Pictures: a path becomes RGB pixels before augmentation; the pixel pipeline makes the tensor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from torch import Tensor

from src.core import Axis, Geometry, InputInfo, Modality, Normalization, TensorShape
from src.data.base import InputEncoder
from src.data.encoders.files import FileEncoder
from src.data.registry import input_encoder_registry

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image(path: Path, *, grayscale: bool = False) -> np.ndarray:
    """Decode a file to RGB (or one gray plane); a missing or unreadable file is named."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if image is None:
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        raise ValueError(f"Could not decode image: {path}")
    return image if grayscale else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@input_encoder_registry.register("image")
class ImageEncoder(FileEncoder, InputEncoder):
    """Declares how a picture reaches the model — size, channels, normalization — and reads its file.

    Pixels are moved by the pipeline each stage declares (``configs/transforms``), which interpolates
    this declaration: ``Resize`` to ``image_size``, ``Normalize`` with ``mean``/``std``, ``ToTensorV2``.
    This encoder therefore reads files and states facts; ``encode`` only checks that the pipeline ran,
    so a chain missing its resize or its tensor step is named here rather than at the first matmul.
    """

    geometry: ClassVar[Geometry] = Geometry.IMAGE

    def __init__(
        self,
        image_size: Sequence[int],
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        *,
        root: str | Path | None = None,
        grayscale: bool = False,
    ) -> None:
        super().__init__(root)
        if len(image_size) != 2 or any(side <= 0 for side in image_size):
            raise ValueError(f"image_size is (height, width) of positive sides, got {list(image_size)}.")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.channels = 1 if grayscale else 3
        self.grayscale = grayscale
        self.normalization = Normalization(tuple(float(value) for value in mean), tuple(float(value) for value in std))
        if len(self.normalization.mean) != self.channels:
            plane = "a gray plane" if grayscale else "RGB"
            raise ValueError(f"mean and std need {self.channels} entries each for {plane}.")

    @property
    def info(self) -> InputInfo:
        height, width = self.image_size
        shape = TensorShape(axes=(Axis.CHANNELS, Axis.HEIGHT, Axis.WIDTH), sizes=(self.channels, height, width))
        return InputInfo(shape=shape, modality=Modality.IMAGE, normalization=self.normalization)

    def load(self, value: object) -> np.ndarray:
        return read_image(self.path_of(value), grayscale=self.grayscale)

    def encode(self, value: object) -> Tensor:
        expected = (self.channels, *self.image_size)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected:
            arrived = list(value.shape) if isinstance(value, Tensor | np.ndarray) else type(value).__name__
            raise ValueError(
                f"Image input arrives as {arrived}, not a {list(expected)} tensor. The stage's transforms pipeline "
                "prepares it: end the chain with Resize to image_size, Normalize and ToTensorV2 (configs/transforms)."
            )
        return value
