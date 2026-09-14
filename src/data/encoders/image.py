"""Pictures: a path becomes RGB pixels before augmentation; the pixel pipeline makes the tensor.

The one place OpenCV is asked to decode, and therefore the place its thread pool is settled too.
"""

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
    """Declares how an image reaches the model — size, channels, normalization — and reads its file.

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
        """The tensor the stage's chain made, checked against what this input was declared to be.

        Read off the end rather than demanded whole, because a stage may draw one picture several times
        and hand the sample a stack of views. What an input *is* stays what one view is — the view axis
        rides with the batch axis, which no declaration carries either — so one leading axis is allowed
        and a second is not: that is a shape nothing in this run builds.

        The check is here rather than left to the model because what it catches is a chain that forgot
        its tail: a picture of the wrong size, or one that never crossed into tensors at all.
        """
        expected = (self.channels, *self.image_size)
        views = len(expected) + 1
        if not isinstance(value, Tensor) or value.ndim > views or tuple(value.shape[-len(expected) :]) != expected:
            arrived = list(value.shape) if isinstance(value, Tensor | np.ndarray) else type(value).__name__
            raise ValueError(
                f"Image input arrives as {arrived}, not a {list(expected)} tensor (nor a stack of them). The "
                "stage's transforms pipeline prepares it: end the chain with Resize to image_size, Normalize "
                "and ToTensorV2 (configs/transforms)."
            )
        return value


def single_threaded_cv2(worker: int) -> None:
    """A DataLoader ``worker_init_fn``: one decoding thread per worker, measured 1.5x faster.

    OpenCV sizes its thread pool to the machine and loader workers are processes, so eight workers
    decode on sixty-four threads that contend for the same cores — here the workers *are* the
    parallelism. A run with no workers keeps cv2's own, which is then the only parallelism there is.
    """
    cv2.setNumThreads(0)
