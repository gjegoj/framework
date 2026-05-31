"""Transforms: adapt raw numpy images into model-ready tensors.

``Transform`` is the framework-agnostic port. Two implementations:

- ``BasicTransform`` — dependency-light resize+normalize+to-tensor (cv2/torch
  only). The default that always works.
- ``AlbumentationsTransform`` — adapter over an Albumentations ``Compose`` for
  rich augmentation pipelines. Albumentations is imported lazily (only when its
  builder is called), so a broken/absent install never breaks importing this
  module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from src.core.entities import Sample
from src.core.keys import IMAGE as IMAGE_KEY

if TYPE_CHECKING:
    import albumentations as A


class Transform(ABC):
    """Transforms a sample's raw image input into a model-ready tensor in place."""

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Transform ``sample`` and return it (image input becomes a CHW tensor)."""


class BasicTransform(Transform):
    """Resize + normalize + to-tensor using only cv2/numpy/torch.

    Mirrors Albumentations' ``Resize`` + ``Normalize`` + ``ToTensorV2`` semantics
    (scale to [0, 1], subtract mean, divide by std, return ``CxHxW`` float).

    Parameters:
        image_size (tuple[int, int]): Target ``(height, width)``.
        mean (list[float]): Per-channel normalization mean.
        std (list[float]): Per-channel normalization std.
    """

    def __init__(self, image_size: tuple[int, int], mean: list[float], std: list[float]) -> None:
        self._height, self._width = image_size
        self._mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self._std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def apply(self, sample: Sample) -> Sample:
        image = sample.inputs[IMAGE_KEY]
        image = cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = (image - self._mean) / self._std
        sample.inputs[IMAGE_KEY] = torch.from_numpy(image)
        return sample


class AlbumentationsTransform(Transform):
    """Adapter over an Albumentations ``Compose`` (built via the lazy builder).

    Parameters:
        compose (A.Compose): Albumentations pipeline ending in ``ToTensorV2``.
    """

    def __init__(self, compose: A.Compose) -> None:
        self._compose = compose

    def apply(self, sample: Sample) -> Sample:
        result = self._compose(image=sample.inputs[IMAGE_KEY])
        sample.inputs[IMAGE_KEY] = result[IMAGE_KEY]
        return sample


def build_basic_transform(
    image_size: tuple[int, int],
    mean: list[float],
    std: list[float],
) -> BasicTransform:
    """Build the dependency-light resize + normalize + to-tensor transform."""
    return BasicTransform(image_size, mean, std)


def build_albumentations_transform(
    image_size: tuple[int, int],
    mean: list[float],
    std: list[float],
) -> AlbumentationsTransform:
    """Build a basic Albumentations pipeline (imported lazily).

    Parameters:
        image_size (tuple[int, int]): Target ``(height, width)``.
        mean (list[float]): Per-channel normalization mean.
        std (list[float]): Per-channel normalization std.

    Returns:
        AlbumentationsTransform: Wrapped Albumentations pipeline.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    height, width = image_size
    compose = A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ]
    )
    return AlbumentationsTransform(compose)
