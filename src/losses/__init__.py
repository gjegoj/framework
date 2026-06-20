"""Losses: criterion bricks behind the ``Criterion`` port.

Importing this package registers every built-in criterion in the ``criteria``
registry — the standard losses (criterion.py) and the metric-learning ones
(angular / contrastive / ranking) alike — so ``import src.losses`` is enough.
"""

from src.losses.angular import ArcFaceCriterion
from src.losses.contrastive import InfoNCECriterion, SigLIPCriterion
from src.losses.criterion import (
    BCEWithLogitsCriterion,
    CrossEntropyCriterion,
    DiceCriterion,
    L1Criterion,
    MSECriterion,
    WeightedSumCriterion,
)
from src.losses.focal import FocalCriterion, FocalLoss
from src.losses.ranking import MarginRankingCriterion, TripletMarginCriterion
from src.losses.registry import criteria

__all__ = [
    "ArcFaceCriterion",
    "BCEWithLogitsCriterion",
    "CrossEntropyCriterion",
    "DiceCriterion",
    "FocalCriterion",
    "FocalLoss",
    "InfoNCECriterion",
    "L1Criterion",
    "MSECriterion",
    "MarginRankingCriterion",
    "SigLIPCriterion",
    "TripletMarginCriterion",
    "WeightedSumCriterion",
    "criteria",
]
