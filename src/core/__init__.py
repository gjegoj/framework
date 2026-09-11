"""Shared values and the registry; the standard library and PyTorch only."""

from __future__ import annotations

from src.core.entities import (
    Batch,
    DatasetInfo,
    InputInfo,
    LossOutput,
    ModelOutput,
    Normalization,
    Prediction,
    Sample,
    StepOutput,
    TargetInfo,
    validate_classes,
    validate_name,
)
from src.core.registry import Registry
from src.core.taxonomy import Axis, Direction, Geometry, Modality, Stage, Stream
from src.core.types import (
    CLASS_AXIS,
    ShapeTree,
    TensorShape,
    TensorTree,
    drop_class_axis,
    require_shape,
    require_tensor,
    tree_map,
)

__all__ = [
    "CLASS_AXIS",
    "Axis",
    "Batch",
    "DatasetInfo",
    "Direction",
    "Geometry",
    "InputInfo",
    "LossOutput",
    "Modality",
    "ModelOutput",
    "Normalization",
    "Prediction",
    "Registry",
    "Sample",
    "ShapeTree",
    "Stage",
    "StepOutput",
    "Stream",
    "TargetInfo",
    "TensorShape",
    "TensorTree",
    "drop_class_axis",
    "require_shape",
    "require_tensor",
    "tree_map",
    "validate_classes",
    "validate_name",
]
