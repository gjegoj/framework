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
"""How ``additional_targets`` names each geometry; ``BOXES`` travels as the pipeline's own
``bboxes`` argument instead. Spelled out because albumentations' ``Targets`` enum is not a
``StrEnum`` and does not satisfy the ``dict[str, str]`` the API declares."""

_BBOXES = "bboxes"
"""Albumentations' own argument for the boxes array, verbatim."""

_BOX_LABELS = "box_labels"
"""Our label field for the boxes' class names: albumentations filters it in step with the
boxes, so a crop that drops a box drops its name."""


class AlbumentationsTransform:
    """Runs one albumentations pipeline over a sample's images and geometric targets.

    One pipeline call, so every sampled parameter is shared between the image, its mask, its
    boxes and every other declared input; undeclared values never reach it. A ``BOXES``
    target travels as ``(float32 [N, 4] xyxy pixels, list of names)``, split into
    ``bboxes`` and a label field and put back together.

    Parameters:
        transforms (Sequence): Albumentations operations, in order; end with ``ToTensorV2``.
        inputs (Mapping[str, Geometry | str]): Inputs to carry, each with its geometry —
            derived at assembly from the loaders, never written by hand.
        targets (Mapping[str, Geometry | str]): Targets to carry, each with its geometry —
            derived at assembly from the encoders.
        auxiliary_inputs (Mapping[str, Geometry | str]): Arrays only the augmentations read.
        label_targets (Sequence[str]): Targets an augmentation may rewrite (a rotation class).
        min_box_visibility (float): Fraction of a box that must survive a crop to be kept.
        min_box_area (float): Same, in pixels. Both are refused without a BOXES target.
        **compose_options (Any): Forwarded verbatim to ``albumentations.Compose`` (``seed``,
            ``p``, ``keypoint_params``, ``strict``); ``telemetry`` defaults to off.
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
        _refuse_a_colliding_name(self._inputs, self._auxiliary_inputs, declared_targets, label_targets)
        _refuse_a_non_pixel_input(self._inputs, self._auxiliary_inputs)
        self._boxes_target = _boxes_target(declared_targets)
        # Every target but the boxes one: those travel as `bboxes`, not as an additional target.
        self._targets = {
            **{name: geometry for name, geometry in declared_targets.items() if name != self._boxes_target},
            **dict.fromkeys(label_targets, Geometry.NONE),
        }
        _refuse_a_derived_option(self._boxes_target, compose_options)
        carried = {**self._inputs, **self._auxiliary_inputs, **self._targets}
        self._pipeline = A.Compose(
            list(transforms),
            additional_targets={name: _PIPELINE_KIND[geometry] for name, geometry in carried.items()},
            **_box_params(self._boxes_target, min_box_visibility, min_box_area),
            # Off by default — a training run should not phone home — but not ours to force.
            **{"telemetry": False, **compose_options},
        )

    def __call__(self, sample: Sample) -> Sample:
        roles = (
            (self._inputs, sample.inputs),
            (self._auxiliary_inputs, sample.auxiliary_inputs),
            (self._targets, sample.targets),
        )
        arguments: dict[str, Any] = {name: values[name] for names, values in roles for name in names}
        if self._boxes_target is not None:
            arguments[_BBOXES], arguments[_BOX_LABELS] = sample.targets[self._boxes_target]
        augmented = self._pipeline(**arguments)
        # Auxiliary inputs are written back too, so a later transform in a chain reads the
        # geometry the image now has; collation never looks at them either way.
        for names, values in roles:
            for name in names:
                values[name] = augmented[name]
        if self._boxes_target is not None:
            # Measured on albumentationsx 2.3.7: bboxes come back as float64 [N, 4], (0, 4) included.
            sample.targets[self._boxes_target] = (augmented[_BBOXES].astype(np.float32), augmented[_BOX_LABELS])
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


def _boxes_target(targets: Mapping[str, Geometry]) -> str | None:
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


def _refuse_a_derived_option(boxes_target: str | None, compose_options: Mapping[str, Any]) -> None:
    """An option this seam derives, declared a second time in ``compose_options``."""
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


def _box_params(boxes_target: str | None, min_visibility: float, min_area: float) -> dict[str, Any]:
    """``bbox_params`` for the one boxes target, or nothing at all; box knobs with no boxes are refused.

    ``pascal_voc`` is albumentations' name for xyxy pixels — the convention ``Instances``
    pins, so no dialect is converted anywhere between the encoder and the batch.
    """
    if boxes_target is None:
        if min_visibility or min_area:
            raise ValueError(
                "min_box_visibility/min_box_area declared, but no target has "
                f"'{Geometry.BOXES}' geometry — the filter would silently never run."
            )
        return {}
    return {
        "bbox_params": A.BboxParams(
            coord_format="pascal_voc",
            label_fields=[_BOX_LABELS],
            min_visibility=min_visibility,
            min_area=min_area,
        )
    }


def _refuse_a_colliding_name(*roles: Sequence[str] | Mapping[str, Geometry]) -> None:
    """A name declared twice, or one albumentations reads as its own, would silently overwrite a value.

    The four roles are separate namespaces everywhere else: an input and a task may
    legally share a name, and only here — as kwargs of one pipeline call — do they collide.
    """
    declared = [name for role in roles for name in role]
    duplicated = sorted({name for name in declared if declared.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"{', '.join(duplicated)}: declared under more than one of 'inputs', 'auxiliary_inputs', "
            "'targets', 'label_targets'. Every name becomes one argument of one pipeline call, so a "
            "duplicate would silently overwrite a value."
        )
    reserved = sorted({name for name in declared if name in {_BBOXES, _BOX_LABELS}})
    if reserved:
        raise ValueError(f"{', '.join(reserved)}: reserved for this seam's boxes arguments. Rename the declared value.")
