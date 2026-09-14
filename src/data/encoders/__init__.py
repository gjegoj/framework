"""Encoders by modality and target type; each registers under the name a config writes."""

from __future__ import annotations

from src.data.encoders.continuous import GaussianBinsEncoder, LinearBinsEncoder, ScalarEncoder
from src.data.encoders.identity import IdentityEncoder
from src.data.encoders.image import ImageEncoder, single_threaded_cv2
from src.data.encoders.label import LabelEncoder, MultilabelEncoder
from src.data.encoders.mask import MaskEncoder
from src.data.encoders.text import TextEncoder

__all__ = [
    "GaussianBinsEncoder",
    "IdentityEncoder",
    "ImageEncoder",
    "LabelEncoder",
    "LinearBinsEncoder",
    "MaskEncoder",
    "MultilabelEncoder",
    "ScalarEncoder",
    "TextEncoder",
    "single_threaded_cv2",
]
