"""Losses: criterion bricks behind the ``Criterion`` port.

Importing this package registers every built-in criterion in the ``criteria``
registry, so ``import src.losses`` is enough. One module per loss family:
``classification`` (cross-entropy / BCE / focal), ``regression`` (MSE / L1),
``segmentation`` (Dice), ``composite`` (weighted sum), and the metric-learning
families ``angular`` (ArcFace), ``contrastive`` (InfoNCE / SigLIP), ``ranking``
(triplet / margin-ranking). ``base.SingleTermCriterion`` is the extension point
for wrapping any single loss module.
"""

from src.losses.angular import ArcFaceCriterion, ProxyAngularCriterion
from src.losses.base import SingleTermCriterion
from src.losses.classification import BCECriterion, CrossEntropyCriterion, FocalCriterion, FocalLoss
from src.losses.composite import WeightedSumCriterion
from src.losses.contrastive import InfoNCECriterion, PairedStreamCriterion, SigLIPCriterion
from src.losses.ranking import MarginRankingCriterion, RankNetCriterion, TripletMarginCriterion
from src.losses.registry import criteria
from src.losses.regression import L1Criterion, MSECriterion
from src.losses.segmentation import DiceCriterion

__all__ = [
    "ArcFaceCriterion",
    "BCECriterion",
    "CrossEntropyCriterion",
    "DiceCriterion",
    "FocalCriterion",
    "FocalLoss",
    "InfoNCECriterion",
    "L1Criterion",
    "MSECriterion",
    "MarginRankingCriterion",
    "PairedStreamCriterion",
    "RankNetCriterion",
    "ProxyAngularCriterion",
    "SigLIPCriterion",
    "SingleTermCriterion",
    "TripletMarginCriterion",
    "WeightedSumCriterion",
    "criteria",
]
