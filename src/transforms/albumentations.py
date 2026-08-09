"""Albumentations behind the ``SampleTransform`` seam."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import albumentations as A

from src.core.taxonomy import Modality

if TYPE_CHECKING:
    from src.core.entities import Sample

IMAGE_TARGET = "image"
MASK_TARGET = "mask"
LABEL_TARGET = "label"
"""How ``additional_targets`` names the three kinds of value a pipeline may carry.

Spelled out rather than taken from albumentations' ``Targets`` enum, which is a
plain ``Enum`` and not a ``StrEnum``: its members are accepted at runtime but do
not satisfy the ``dict[str, str]`` the API declares.
"""


class AlbumentationsTransform:
    """Runs one albumentations pipeline over a sample's images and spatial targets.

    Everything travels through a *single* pipeline call, so each sampled
    parameter is shared: the crop taken from the image is the same crop taken
    from its mask and from every other declared image.

    Only declared values are handed to the pipeline. Undeclared ones — an
    embedding, a class label no augmentation is about — never reach it, so they
    cannot be touched even when their name happens to be one albumentations
    reserves, such as ``mask``.

    The pipeline is built here rather than accepted ready-made: registering
    the extra keys must not mutate a pipeline the caller may reuse for
    another schema.

    Independent sampling per view is composition, not a flag: wrap this
    transform in ``MultiViewTransform`` to augment each view afresh.

    Parameters:
        transforms (Sequence): Albumentations operations, in order; end with
            ``ToTensorV2`` to hand the model tensors. Named as ``Compose`` names
            it, like every other argument here.
        image_inputs (Sequence[str]): Sample inputs holding images; every one
            of them is registered as an image target under its own name.
        spatial_targets (Sequence[str]): Targets that follow the image's
            geometry — the ones whose encoder is ``spatial``.
        label_targets (Sequence[str]): Targets an augmentation may rewrite — the
            rotation class a quarter-turn advances, the flag a crop sets. Unlike
            spatial targets these are not derived from the encoders: a rotation
            label and a class label are both plain labels, and only the
            experiment knows which one the pipeline's augmentations are about.
            Every key declared here is rewritten by every augmentation that has
            an ``apply_to_label``.
        **compose_options (Any): Forwarded verbatim to ``albumentations.Compose``,
            so every knob it has stays reachable from config without a change
            here. ``seed`` for a reproducible pipeline, ``p`` for the chance the
            whole thing applies, ``bbox_params`` / ``keypoint_params`` (plain
            mappings) for boxes and keypoints, ``strict``, ``is_check_shapes``,
            ``mask_interpolation``. ``telemetry`` defaults to off and can be
            turned back on.
    """

    def __init__(
        self,
        transforms: Sequence[Any],
        image_inputs: Sequence[str] = (Modality.IMAGE,),
        spatial_targets: Sequence[str] = (),
        label_targets: Sequence[str] = (),
        **compose_options: Any,
    ) -> None:
        if not image_inputs:
            raise ValueError("AlbumentationsTransform needs at least one image input.")
        if "additional_targets" in compose_options:
            raise ValueError(
                "'additional_targets' is derived here, from 'image_inputs', 'spatial_targets' and "
                "'label_targets'. Declare those instead, so the keys a pipeline registers cannot "
                "contradict the values it is given."
            )
        self._image_inputs = list(image_inputs)
        self._targets = [*spatial_targets, *label_targets]
        self._pipeline = A.Compose(
            list(transforms),
            additional_targets={
                **dict.fromkeys(self._image_inputs, IMAGE_TARGET),
                **dict.fromkeys(spatial_targets, MASK_TARGET),
                **dict.fromkeys(label_targets, LABEL_TARGET),
            },
            # Off by default — a training run should not phone home — but not ours to force.
            **{"telemetry": False, **compose_options},
        )

    def __call__(self, sample: Sample) -> Sample:
        augmented = self._pipeline(
            **{name: sample.inputs[name] for name in self._image_inputs},
            **{name: sample.targets[name] for name in self._targets},
        )
        for name in self._image_inputs:
            sample.inputs[name] = augmented[name]
        for name in self._targets:
            sample.targets[name] = augmented[name]
        return sample
