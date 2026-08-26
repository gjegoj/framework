"""Target encoders: raw column values into the values a task's target starts as."""

from __future__ import annotations

from src.data.encoders.base import TargetEncoder
from src.data.encoders.boxes import BoxesTargetEncoder
from src.data.encoders.continuous import (
    BinnedTargetEncoder,
    GaussianBinsTargetEncoder,
    LinearBinsTargetEncoder,
    ScalarTargetEncoder,
)
from src.data.encoders.label import LabelTargetEncoder, MultiLabelTargetEncoder
from src.data.encoders.mask import MaskTargetEncoder

__all__ = [
    "BinnedTargetEncoder",
    "BoxesTargetEncoder",
    "GaussianBinsTargetEncoder",
    "LabelTargetEncoder",
    "LinearBinsTargetEncoder",
    "MaskTargetEncoder",
    "MultiLabelTargetEncoder",
    "ScalarTargetEncoder",
    "TargetEncoder",
]
