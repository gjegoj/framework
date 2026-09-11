"""Task semantics: what a run learns for one target, and what it declares to have assembled around it.

Importing this package is what makes its kinds resolvable: `tasks.<name>.kind: classification` finds one here.
"""

from __future__ import annotations

from src.tasks.base import Task
from src.tasks.classification import BinaryClassification, Classification, MultilabelClassification
from src.tasks.labels import BinaryLabels, MulticlassLabels, MultilabelLabels
from src.tasks.regression import Regression
from src.tasks.segmentation import BinarySegmentation, DenseOutput, Segmentation

__all__ = [
    "BinaryClassification",
    "BinaryLabels",
    "BinarySegmentation",
    "Classification",
    "DenseOutput",
    "MulticlassLabels",
    "MultilabelClassification",
    "MultilabelLabels",
    "Regression",
    "Segmentation",
    "Task",
]
