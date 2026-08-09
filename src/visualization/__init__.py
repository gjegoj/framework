"""The visualization capability: model outputs drawn over the inputs that made them."""

from __future__ import annotations

from src.visualization.annotators import Annotator, build_annotators
from src.visualization.entities import (
    KINDS,
    Classification,
    Classifications,
    Image,
    Kind,
    Label,
    Media,
    Regression,
    SampleView,
    Score,
    Segmentation,
    SegmentationClass,
    Text,
    Verdict,
)
from src.visualization.fields import MAX_CHIP_CHARS
from src.visualization.html import MAX_DISPLAY_SIDE, HtmlRenderer

__all__ = [
    "KINDS",
    "MAX_CHIP_CHARS",
    "MAX_DISPLAY_SIDE",
    "Annotator",
    "Classification",
    "Classifications",
    "HtmlRenderer",
    "Image",
    "Kind",
    "Label",
    "Media",
    "Regression",
    "SampleView",
    "Score",
    "Segmentation",
    "SegmentationClass",
    "Text",
    "Verdict",
    "build_annotators",
]
