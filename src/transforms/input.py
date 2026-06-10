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
        sample.inputs = {k: result[k] for k in sample.inputs}
        sample.targets = {k: result[k] for k in sample.targets}
        return sample
