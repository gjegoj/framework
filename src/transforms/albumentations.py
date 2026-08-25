"""Albumentations behind the ``SampleTransform`` seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import albumentations as A
import numpy as np

from src.core.taxonomy import Geometry, Modality

if TYPE_CHECKING:
    from src.core.entities import Sample

_PIPELINE_KIND = {Geometry.IMAGE: "image", Geometry.MASK: "mask", Geometry.NONE: "label"}
"""How ``additional_targets`` names each geometry a value may ride under.

Spelled out rather than taken from albumentations' ``Targets`` enum, which is a plain
``Enum`` and not a ``StrEnum``: its members are accepted at runtime but do not satisfy
the ``dict[str, str]`` the API declares. ``BOXES`` is absent on purpose — boxes ride the
pipeline's own ``bboxes`` argument, not an additional target: measured on albumentationsx
2.3.7, ``label_fields`` are not plumbed through ``additional_targets`` and a second boxes
field raises ``KeyError`` on its labels.
"""

BOXES = "bboxes"
"""Albumentations' own argument for the boxes array — its name, spelled once here."""

BOX_LABELS = "box_labels"
"""The label field a BOXES target's class names travel under inside the pipeline call.

Their own key rather than the target's name, because albumentations filters this list in
step with the boxes — a crop that drops a box drops its name — and the pair is put back
together under the target's name afterwards.
"""


class AlbumentationsTransform:
    """Runs one albumentations pipeline over a sample's images and geometric targets.

    Everything travels through a *single* pipeline call, so each sampled
    parameter is shared: the crop taken from the image is the same crop taken
    from its mask, its boxes and every other declared image.

    Only declared values are handed to the pipeline. Undeclared ones — an
    embedding, a class label no augmentation is about — never reach it, so they
    cannot be touched even when their name happens to be one albumentations
    reserves, such as ``mask``.

    The pipeline is built here rather than accepted ready-made: registering
    the extra keys must not mutate a pipeline the caller may reuse for
    another schema.

    Independent sampling per view is composition, not a flag: wrap this
    transform in ``MultiViewTransform`` to augment each view afresh.

    A ``Geometry.BOXES`` target travels as the pair its geometry documents —
    ``(float32 [N, 4] xyxy-pixel array, list of class names)`` — split into the
    pipeline's ``bboxes`` and ``BOX_LABELS`` arguments and put back together
    afterwards, so what enters and leaves this seam is one value under one name.

    Parameters:
        transforms (Sequence): Albumentations operations, in order; end with
            ``ToTensorV2`` to hand the model tensors. Named as ``Compose`` names
            it, like every other argument here.
        inputs (Mapping[str, Geometry | str]): Sample inputs to carry, each beside
            how it rides geometry: ``IMAGE`` for light (interpolated, normalised),
            ``MASK`` for per-pixel labels the model consumes beside the image
            (nearest-neighbour, untouched by ``Normalize``). Derived at assembly
            from each input's loader; never written by hand in config.
        targets (Mapping[str, Geometry | str]): Task targets to carry, each beside
            its geometry — a mask target, a boxes target. Derived at assembly from
            each task's encoder, so a target cannot fall out of step with its image.
        auxiliary_inputs (Mapping[str, Geometry | str]): The sample's auxiliary
            inputs to carry — arrays only the augmentations read. Derived at
            assembly from ``data.auxiliary_inputs`` and their loaders.
        label_targets (Sequence[str]): Targets an augmentation may rewrite — the
            rotation class a quarter-turn advances, the flag a crop sets. The one
            hand-declared role here: a rotation label and a class label are both
            plain labels, and only the experiment knows which one the pipeline's
            augmentations are about. Every key declared here is rewritten by every
            augmentation that has an ``apply_to_label``.
        min_box_visibility (float): Fraction of a box that must survive a crop for
            the box — and its name — to be kept. Zero keeps whatever albumentations
            returns.
        min_box_area (float): Same, in pixels of area. Both forward into
            ``BboxParams`` and are refused without a BOXES target, where they would
            read as filtering that never ran.
        **compose_options (Any): Forwarded verbatim to ``albumentations.Compose``,
            so every knob it has stays reachable from config without a change
            here. ``seed`` for a reproducible pipeline, ``p`` for the chance the
            whole thing applies, ``keypoint_params`` (a plain mapping) for
            keypoints, ``strict``, ``is_check_shapes``, ``mask_interpolation``.
            ``telemetry`` defaults to off and can be turned back on.
    """

    def __init__(
        self,
        transforms: Sequence[Any],
        inputs: Mapping[str, Geometry | str] | None = None,
        targets: Mapping[str, Geometry | str] | None = None,
        auxiliary_inputs: Mapping[str, Geometry | str] | None = None,
        label_targets: Sequence[str] = (),
        min_box_visibility: float = 0.0,
        min_box_area: float = 0.0,
        **compose_options: Any,
    ) -> None:
        self._inputs = _geometries("inputs", inputs if inputs is not None else {Modality.IMAGE: Geometry.IMAGE})
        self._auxiliary_inputs = _geometries("auxiliary_inputs", auxiliary_inputs or {})
        declared_targets = _geometries("targets", targets or {})
        if not self._inputs:
            raise ValueError("AlbumentationsTransform needs at least one image input.")
        _refuse_a_name_in_two_roles(self._inputs, self._auxiliary_inputs, declared_targets, label_targets)
        _refuse_a_non_pixel_input(self._inputs, self._auxiliary_inputs)
        self._targets = {**declared_targets, **dict.fromkeys(label_targets, Geometry.NONE)}
        self._boxes_target = _the_one_boxes_target(self._targets)
        _refuse_contradicted_options(self._boxes_target, compose_options, min_box_visibility, min_box_area)
        carried = {**self._inputs, **self._auxiliary_inputs, **self._targets}
        self._pipeline = A.Compose(
            list(transforms),
            additional_targets={
                name: _PIPELINE_KIND[geometry] for name, geometry in carried.items() if name != self._boxes_target
            },
            # Off by default — a training run should not phone home — but not ours to force.
            **_box_params(self._boxes_target, min_box_visibility, min_box_area),
            **{"telemetry": False, **compose_options},
        )

    def __call__(self, sample: Sample) -> Sample:
        given: dict[str, Any] = {
            **{name: sample.inputs[name] for name in self._inputs},
            **{name: sample.auxiliary_inputs[name] for name in self._auxiliary_inputs},
            **{name: sample.targets[name] for name in self._targets if name != self._boxes_target},
        }
        if self._boxes_target is not None:
            boxes, names = sample.targets[self._boxes_target]
            given[BOXES], given[BOX_LABELS] = boxes, list(names)
        augmented = self._pipeline(**given)
        for name in self._inputs:
            sample.inputs[name] = augmented[name]
        # Written back so a later transform in a chain reads the geometry the image now
        # has; collation never looks at the field either way.
        for name in self._auxiliary_inputs:
            sample.auxiliary_inputs[name] = augmented[name]
        for name in self._targets:
            if name != self._boxes_target:
                sample.targets[name] = augmented[name]
        if self._boxes_target is not None:
            sample.targets[self._boxes_target] = (
                np.asarray(augmented[BOXES], dtype=np.float32).reshape(-1, 4),
                list(augmented[BOX_LABELS]),
            )
        return sample


def _geometries(role: str, declared: Mapping[str, Geometry | str]) -> dict[str, Geometry]:
    """Declared kinds as members, refusing an unknown spelling by role and value.

    A mapping reaching this seam from config holds plain strings; one built by assembly
    holds members. Normalising here is what lets both be written the natural way.
    """
    members = set(Geometry)
    known = ", ".join(Geometry)
    for name, value in declared.items():
        if value not in members:
            raise ValueError(f"'{name}' in {role} declares geometry '{value}'. Known geometries: {known}.")
    return {name: Geometry(value) for name, value in declared.items()}


def _refuse_a_non_pixel_input(*roles: Mapping[str, Geometry]) -> None:
    """An input reaching the pipeline is pixels; anything else is a declaration error.

    A ``NONE`` input would be handed to albumentations as a label, and a ``BOXES`` one
    has no image to belong to — both are mistakes worth naming while the experiment is
    built rather than shapes to guess at every epoch. Assembly filters ``NONE`` columns
    out before this seam, so what arrives here arrived by hand.
    """
    for declared in roles:
        for name, geometry in declared.items():
            if geometry not in (Geometry.IMAGE, Geometry.MASK):
                raise ValueError(
                    f"Input '{name}' declares geometry '{geometry}', but a sample input the pipeline "
                    f"carries is pixels — '{Geometry.IMAGE}' or '{Geometry.MASK}'."
                )


def _the_one_boxes_target(targets: Mapping[str, Geometry]) -> str | None:
    """The single BOXES target, or ``None``; two are refused naming both.

    Measured on albumentationsx 2.3.7: a second boxes field registered through
    ``additional_targets`` raises ``KeyError`` on its label field, so its names would
    not be filtered with its boxes. Refusing beats corrupting.
    """
    boxed = sorted(name for name, geometry in targets.items() if geometry is Geometry.BOXES)
    if len(boxed) > 1:
        raise ValueError(
            f"Targets '{"', '".join(boxed)}' both declare '{Geometry.BOXES}' geometry, but one pipeline "
            f"carries one boxes field (measured: albumentationsx 2.3.7 does not plumb label fields "
            f"through additional targets). Declare one boxes task per run."
        )
    return boxed[0] if boxed else None


def _refuse_contradicted_options(
    boxes_target: str | None, compose_options: Mapping[str, Any], min_visibility: float, min_area: float
) -> None:
    """Options this seam derives, declared a second time — or box knobs with no boxes."""
    if "additional_targets" in compose_options:
        raise ValueError(
            "'additional_targets' is derived here, from 'inputs', 'targets', 'auxiliary_inputs' and "
            "'label_targets'. Declare those instead, so the keys a pipeline registers cannot "
            "contradict the values it is given."
        )
    if boxes_target is not None and "bbox_params" in compose_options:
        raise ValueError(
            f"'bbox_params' is derived from the '{boxes_target}' target — pascal_voc pixels, the label "
            f"field, and the min_box knobs. Declare those instead."
        )
    if boxes_target is None and (min_visibility or min_area):
        raise ValueError(
            "min_box_visibility/min_box_area declared, but no target rides "
            f"'{Geometry.BOXES}' geometry — the filter would silently never run."
        )


def _box_params(boxes_target: str | None, min_visibility: float, min_area: float) -> dict[str, Any]:
    """``bbox_params`` for the one boxes target, or nothing at all.

    ``pascal_voc`` is albumentations' name for xyxy pixels — the convention ``Instances``
    pins, so no dialect is converted anywhere between the encoder and the batch.
    """
    if boxes_target is None:
        return {}
    return {
        "bbox_params": A.BboxParams(
            coord_format="pascal_voc",
            label_fields=[BOX_LABELS],
            min_visibility=min_visibility,
            min_area=min_area,
        )
    }


def _refuse_a_name_in_two_roles(*roles: Sequence[str] | Mapping[str, Geometry]) -> None:
    """A name declared twice would be one kwarg of one pipeline call — a silent overwrite.

    The four roles are separate namespaces everywhere else: an input and a task may
    legally share a name, and only here do they collide. Naming the clash at
    construction beats one value quietly winning over another every epoch.
    """
    declared = [name for role in roles for name in role]
    duplicated = sorted({name for name in declared if declared.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"{', '.join(duplicated)}: declared under more than one of 'inputs', 'auxiliary_inputs', "
            "'targets', 'label_targets'. Every name becomes one argument of one pipeline call, so a "
            "duplicate would silently overwrite a value."
        )
    reserved = sorted({name for name in declared if name in {BOXES, BOX_LABELS}})
    if reserved:
        raise ValueError(f"{', '.join(reserved)}: reserved for this seam's boxes carrier. Rename the declared value.")
