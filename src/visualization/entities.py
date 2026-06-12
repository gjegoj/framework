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


Label = Classification | Classifications


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
