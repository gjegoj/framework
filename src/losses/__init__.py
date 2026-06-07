"""Losses: criterion bricks behind the ``Criterion`` port."""

from src.losses.criterion import (
    BCEWithLogitsCriterion,
    CrossEntropyCriterion,
    DiceCriterion,
    L1Criterion,
    MSECriterion,
    WeightedSumCriterion,
    criteria,
)

__all__ = [
    "BCEWithLogitsCriterion",
    "WeightedSumCriterion",
    "CrossEntropyCriterion",
    "DiceCriterion",
    "L1Criterion",
    "MSECriterion",
    "criteria",
]
