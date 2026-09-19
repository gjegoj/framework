"""Objectives: what a task is learned by, and how several of them combine.

Importing this package is what makes its names resolvable: `loss: dice` in a config finds one here.
"""

from __future__ import annotations

from src.losses.base import Loss, NamedLoss, TorchLoss
from src.losses.classification import BinaryCrossEntropy, CrossEntropy, Focal
from src.losses.composite import WeightedSum
from src.losses.contrastive import InfoNce
from src.losses.distillation import KullbackLeibler
from src.losses.metric_learning import ArcFace, ArcFaceProxy
from src.losses.regression import Expectation, Huber, MeanAbsoluteError, MeanSquaredError, SmoothL1
from src.losses.segmentation import Dice, IntersectionOverUnion, Tversky

__all__ = [
    "ArcFace",
    "ArcFaceProxy",
    "BinaryCrossEntropy",
    "CrossEntropy",
    "Dice",
    "Expectation",
    "Focal",
    "Huber",
    "InfoNce",
    "IntersectionOverUnion",
    "KullbackLeibler",
    "Loss",
    "MeanAbsoluteError",
    "MeanSquaredError",
    "NamedLoss",
    "SmoothL1",
    "TorchLoss",
    "Tversky",
    "WeightedSum",
]
