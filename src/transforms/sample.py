"""Per-sample input transforms: adapt raw numpy images into model-ready tensors.

``Transform`` is the framework-agnostic port. ``AlbumentationsTransform`` wraps an
Albumentations ``Compose``. Spatial targets (segmentation masks) are registered via
``add_targets`` at construction so they ride through the same geometric pipeline as
the image — the same pattern used in the old prototype.

Augmentation pipelines are declared in YAML using ``_target_``-keyed specs and
instantiated via ``src.core.instantiate.instantiate``.  See
``configs/transforms/`` for examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import albumentations as A
import torch

from src.core.entities import Sample


class Transform(ABC):
    """Transforms a sample's raw image input (and any spatial targets) into tensors."""

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Transform ``sample`` in place and return it (inputs/targets become tensors)."""


class AlbumentationsTransform(Transform):
    """Adapter over an Albumentations ``Compose``.

    Spatial targets (mask arrays already in ``sample.targets``) are passed
    alongside the image via ``**sample.inputs, **sample.targets`` in a single
    call, so every geometric operation is applied consistently to all of them.

    ``spatial_targets`` tells Albumentations which target keys are masks so it
    applies nearest-neighbour resizing and correct dtypes to them.

    Parameters:
        compose (A.Compose): Albumentations pipeline ending in ``ToTensorV2``.
        spatial_targets (list[str] | None): Target key names that are spatial (masks).
            Registered as ``"mask"`` type via ``add_targets``.
    """

    def __init__(self, compose: A.Compose, spatial_targets: list[str] | None = None) -> None:
        self._compose = compose
        if spatial_targets:
            self._compose.add_targets({name: "mask" for name in spatial_targets})

    def apply(self, sample: Sample) -> Sample:
        result = self._compose(**sample.inputs, **sample.targets)
        sample.inputs = {key: result[key] for key in sample.inputs}
        sample.targets = {key: result[key] for key in sample.targets}
        return sample


class IdentityTransform(Transform):
    """Pass-through transform for inputs that are already model-ready vectors.

    Converts each input array to a float ``torch.Tensor`` (so collation can stack
    it into ``[B, D]``) and leaves targets untouched.  Used for the embedding
    modality, where the input is a precomputed ``[D]`` vector and the Albumentations
    image pipeline does not apply; targets are tensorised by their ``TargetEncoder``,
    exactly as in the image flow.
    """

    def apply(self, sample: Sample) -> Sample:
        sample.inputs = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in sample.inputs.items()}
        return sample


class MultiViewTransform(Transform):
    """Applies an Albumentations pipeline to each image input independently.

    Single-view experiments have one ``"image"`` key and ``AlbumentationsTransform``
    handles them directly. Multi-view experiments (ranking: anchor / positive /
    negative) have several image keys; this wrapper applies the same pipeline to
    each independently, collecting results back under their original keys.

    Spatial targets (masks) are not supported here — ranking tasks carry no dense
    targets so the simpler ``image=`` call suffices.

    Parameters:
        compose (A.Compose): Albumentations pipeline ending in ``ToTensorV2``.

    Streams:
        Inputs (any keys) → tensors of shape ``[C, H, W]`` after ``ToTensorV2``.
    """

    def __init__(self, compose: A.Compose) -> None:
        self._compose = compose

    def apply(self, sample: Sample) -> Sample:
        sample.inputs = {key: self._compose(image=value)["image"] for key, value in sample.inputs.items()}
        return sample
