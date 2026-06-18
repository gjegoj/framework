"""Losses: criterion bricks behind the ``Criterion`` port."""

from src.losses.criterion import (
    BCEWithLogitsCriterion,
    CrossEntropyCriterion,
    DiceCriterion,
    L1Criterion,
    MSECriterion,
    WeightedSumCriterion,
)
from src.losses.registry import criteria

__all__ = [
    "BCEWithLogitsCriterion",
    "WeightedSumCriterion",
    "CrossEntropyCriterion",
    "DiceCriterion",
    "L1Criterion",
    "MSECriterion",
    "criteria",
]
