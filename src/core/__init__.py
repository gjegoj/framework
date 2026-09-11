"""Shared values and the registry; the standard library and PyTorch only."""

from __future__ import annotations

from src.core.entities import (
    SEGMENT,
    SPLIT,
    Batch,
    DatasetInfo,
    InputInfo,
    LossOutput,
    Matrix,
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
from src.core.taxonomy import Axis, Geometry, Modality, Role, Semantics, Stage, Stream
from src.core.types import (
    CLASS_AXIS,
    ShapeTree,
    TensorShape,
    TensorTree,
    drop_class_axis,
    require_tensor,
    tree_map,
)

__all__ = [
    "CLASS_AXIS",
    "SEGMENT",
    "SPLIT",
    "Axis",
    "Batch",
    "DatasetInfo",
    "Geometry",
    "InputInfo",
    "LossOutput",
    "Matrix",
    "Modality",
    "ModelOutput",
    "Normalization",
    "Prediction",
    "Registry",
    "Role",
    "Sample",
    "Semantics",
    "ShapeTree",
    "Stage",
    "StepOutput",
    "Stream",
    "TargetInfo",
    "TensorShape",
    "TensorTree",
    "drop_class_axis",
    "require_tensor",
    "tree_map",
    "validate_classes",
    "validate_name",
]
