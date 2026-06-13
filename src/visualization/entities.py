"""Visualization IR: a sample projected for display, with FiftyOne-style labels.

These are plain dataclasses (numpy + stdlib only) — the framework-agnostic
intermediate representation between the model-output annotators and the renderer.
``Label`` is a union extended in later phases (``Regression``, ``Segmentation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Classification:
    """A single-label prediction or ground truth.

    Parameters:
        label (str): The class name.
        confidence (float | None): Optional softmax/sigmoid probability.
    """

    label: str
    confidence: float | None = None


@dataclass
class Classifications:
    """A multilabel set of classifications.

    Parameters:
        items (list[Classification]): The active labels (each may carry confidence).
    """

    items: list[Classification] = field(default_factory=list)


@dataclass(frozen=True)
class RegressionComponent:
    """One regressed quantity (a scalar output = a single component, ``name=""``).

    Parameters:
        name (str): Component name (``""`` for a scalar; e.g. ``"height"`` for a vector).
        value (float): The regressed value (ground truth or prediction).
        error (float | None): Signed ``pred - gt``; set by the annotator on the pred side.
    """

    name: str
    value: float
    error: float | None = None


@dataclass
class Regression:
    """A regression prediction or ground truth: one or more named components.

    Parameters:
        components (list[RegressionComponent]): The regressed quantities.
    """

    components: list[RegressionComponent] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentationClass:
    """One class's binary mask within a sample (at display resolution).

    Parameters:
        name (str): Class name.
        mask (np.ndarray): Boolean ``[H, W]`` mask for this class.
    """

    name: str
    mask: np.ndarray


@dataclass
class Segmentation:
    """A segmentation prediction or ground truth: the per-class masks present.

    Parameters:
        classes (list[SegmentationClass]): One entry per class present in the sample.
    """

    classes: list[SegmentationClass] = field(default_factory=list)


Label = Classification | Classifications | Regression | Segmentation


@dataclass
class SampleView:
    """One example projected for visual inspection — image plus named label fields.

    Mirrors the ``TargetView`` / ``TaskStepView`` "View" family: a projection of a
    sample's data for a specific purpose (here, rendering). Each task contributes
    two fields, ``{task}_gt`` and ``{task}_pred``.

    Parameters:
        image (np.ndarray): Display-ready ``[H, W, C]`` uint8 RGB image.
        fields (dict[str, Label]): Named labels (e.g. ``"species_gt"``).
        tags (list[str]): Free-form tags (e.g. ``"species:correct"``).
        metadata (dict[str, Any]): Free-form scalars (index, id, ...).
    """

    image: np.ndarray
    fields: dict[str, Label] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
