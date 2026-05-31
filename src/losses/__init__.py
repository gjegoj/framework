"""Losses: criterion bricks behind the ``Criterion`` port."""

from src.losses.criterion import (
    BCEWithLogitsCriterion,
    CrossEntropyCriterion,
    L1Criterion,
    MSECriterion,
    criteria,
)

__all__ = [
    "BCEWithLogitsCriterion",
    "CrossEntropyCriterion",
    "L1Criterion",
    "MSECriterion",
    "criteria",
]
