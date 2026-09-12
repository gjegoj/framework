"""Task semantics: what a run learns for one target, and what it declares to have assembled around it.

Importing this package is what makes its kinds resolvable: `tasks.<name>.kind: classification` finds one here.
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.classification import BinaryClassification, Classification, MultilabelClassification
from src.tasks.regression import Regression
from src.tasks.segmentation import BinarySegmentation, DenseOutput, Segmentation
from src.tasks.semantics import BinarySemantics, MulticlassSemantics, MultilabelSemantics

__all__ = [
    "BinaryClassification",
    "BinarySegmentation",
    "BinarySemantics",
    "Classification",
    "DenseOutput",
    "MulticlassSemantics",
    "MultilabelClassification",
    "MultilabelSemantics",
    "Regression",
    "Segmentation",
    "Task",
]
