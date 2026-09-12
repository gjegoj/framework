"""The transforms section becomes one callable per stage, and a pixel one is told what moves with it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config import ComponentConfig
from src.core import Geometry, Sample, Stage
from src.transforms.build import build_transforms

GEOMETRIES = {"inputs": {"image": Geometry.IMAGE}, "targets": {"mask": Geometry.MASK}, "auxiliary_inputs": {}}


class Flip:
    """A transform that records what it was bound to, so the binding itself can be read back."""

    bound: Mapping[str, Mapping[str, Geometry]] = {}

    def with_geometry(
        self, inputs: Mapping[str, Geometry], targets: Mapping[str, Geometry], auxiliary_inputs: Mapping[str, Geometry]
    ) -> Any:
        type(self).bound = {
            "inputs": dict(inputs),
            "targets": dict(targets),
            "auxiliary_inputs": dict(auxiliary_inputs),
        }
        return lambda sample: sample

    def __call__(self, sample: Sample) -> Sample:
        return sample


def test_transforms_are_keyed_by_split_and_told_the_geometry() -> None:
    declared = {Stage.TRAIN: ComponentConfig.model_validate({"_target_": "tests.unit.transforms.test_build.Flip"})}

    built = build_transforms(declared, GEOMETRIES)

    assert set(built) == {"train"} and Flip.bound == GEOMETRIES
