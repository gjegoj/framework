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
        mask_inputs (Sequence[str]): Sample inputs whose values are per-pixel labels
            rather than light — a mask the model consumes beside the image. Mask-kind
            in the pipeline (nearest-neighbour geometry, untouched by ``Normalize``)
            and collated like any other input. Derived at assembly from each input's
            loader (``loader: {name: mask}``); never written by hand in config.
        spatial_targets (Sequence[str]): Targets that follow the image's
            geometry — the ones whose encoder is ``spatial``.
        auxiliary_inputs (Sequence[str]): The sample's auxiliary inputs to carry
            through the pipeline — mask-kind, so geometry samples them
            nearest-neighbour and ``Normalize`` never touches them. Derived at
            assembly from ``data.auxiliary_inputs``; written by hand only when this
            transform is built directly.
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
        mask_inputs: Sequence[str] = (),
        spatial_targets: Sequence[str] = (),
        label_targets: Sequence[str] = (),
        auxiliary_inputs: Sequence[str] = (),
        **compose_options: Any,
    ) -> None:
        if not image_inputs:
            raise ValueError("AlbumentationsTransform needs at least one image input.")
        if "additional_targets" in compose_options:
            raise ValueError(
                "'additional_targets' is derived here, from 'image_inputs', 'mask_inputs', "
                "'auxiliary_inputs', 'spatial_targets' and 'label_targets'. Declare those "
                "instead, so the keys a pipeline registers cannot contradict the values it "
                "is given."
            )
        _refuse_a_name_in_two_roles(image_inputs, mask_inputs, auxiliary_inputs, spatial_targets, label_targets)
        # Both are model inputs and both are written back to `sample.inputs`; only their
        # treatment inside the pipeline differs, which is what the two lists are for.
        self._image_inputs = [*image_inputs, *mask_inputs]
        self._auxiliary_inputs = list(auxiliary_inputs)
        self._targets = [*spatial_targets, *label_targets]
        self._pipeline = A.Compose(
            list(transforms),
            additional_targets={
                **dict.fromkeys(image_inputs, IMAGE_TARGET),
                **dict.fromkeys(mask_inputs, MASK_TARGET),
                **dict.fromkeys(self._auxiliary_inputs, MASK_TARGET),
                **dict.fromkeys(spatial_targets, MASK_TARGET),
                **dict.fromkeys(label_targets, LABEL_TARGET),
            },
            # Off by default — a training run should not phone home — but not ours to force.
            **{"telemetry": False, **compose_options},
        )

    def __call__(self, sample: Sample) -> Sample:
        augmented = self._pipeline(
            **{name: sample.inputs[name] for name in self._image_inputs},
            **{name: sample.auxiliary_inputs[name] for name in self._auxiliary_inputs},
            **{name: sample.targets[name] for name in self._targets},
        )
        for name in self._image_inputs:
            sample.inputs[name] = augmented[name]
        # Written back so a later transform in a chain reads the geometry the image now
        # has; collation never looks at the field either way.
        for name in self._auxiliary_inputs:
            sample.auxiliary_inputs[name] = augmented[name]
        for name in self._targets:
            sample.targets[name] = augmented[name]
        return sample


def _refuse_a_name_in_two_roles(*roles: Sequence[str]) -> None:
    """A name declared twice would be one kwarg of one pipeline call — a silent overwrite.

    The four roles are separate namespaces everywhere else: an input and a task may
    legally share a name, and only here do they collide. Naming the clash at
    construction beats one value quietly winning over another every epoch.
    """
    declared = [name for role in roles for name in role]
    duplicated = sorted({name for name in declared if declared.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"{', '.join(duplicated)}: declared under more than one of 'image_inputs', "
            "'mask_inputs', 'auxiliary_inputs', 'spatial_targets', 'label_targets'. Every name "
            "becomes one argument of one pipeline call, so a duplicate would silently overwrite "
            "a value."
        )
