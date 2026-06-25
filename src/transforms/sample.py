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
    """Applies an Albumentations pipeline to several image inputs (ranking / multi-stream views).

    Single-view experiments have one ``"image"`` key and ``AlbumentationsTransform`` handles them
    directly. Multi-view experiments have several image keys; this transform applies the pipeline to
    each, collecting results back under their original keys. Two sampling modes:

    - ``shared=False`` (default): each view is augmented *independently* — the pipeline is sampled
      afresh per view (different crop/flip/jitter each). Right for metric learning / triplet, where
      the backbone should stay invariant to augmentation and a positive may itself be an augmented view.
    - ``shared=True``: *one* sampling is applied jointly to every view (same crop/flip/jitter for all),
      via Albumentations ``additional_targets``. Right for pairwise quality ranking, where differential
      augmentation would inject a spurious signal into the comparison.

    Spatial targets (masks) are not supported here — ranking tasks carry no dense targets.

    Parameters:
        compose (A.Compose): Albumentations pipeline ending in ``ToTensorV2``.
        shared (bool): Apply one sampling jointly to all views (``True``) or sample each view
            independently (``False``, default).

    Streams:
        Inputs (any keys) → tensors of shape ``[C, H, W]`` after ``ToTensorV2``.
    """

    def __init__(self, compose: A.Compose, shared: bool = False) -> None:
        self._compose = compose
        self._shared = shared
        self._registered = False

    def apply(self, sample: Sample) -> Sample:
        if not self._shared:  # independent: re-sample the pipeline per view
            sample.inputs = {key: self._compose(image=value)["image"] for key, value in sample.inputs.items()}
            return sample
        # shared: register every view as an image target once, so a single call applies identical
        # sampled params to all of them.
        if not self._registered:
            self._compose.add_targets({key: "image" for key in sample.inputs if key != "image"})
            self._registered = True
        transformed = self._compose(**sample.inputs)
        sample.inputs = {key: transformed[key] for key in sample.inputs}
        return sample
