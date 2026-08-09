"""The losses capability: implementations of the core ``Criterion`` port, by task family."""

from __future__ import annotations

from src.losses.angular import ArcFaceCriterion, ProxyAngularCriterion
from src.losses.base import WrappedCriterion
from src.losses.classification import BinaryCrossEntropyCriterion, CrossEntropyCriterion, FocalCriterion
from src.losses.composite import WeightedSumCriterion
from src.losses.contrastive import InfoNceCriterion, SigLipCriterion, TripletCriterion
from src.losses.distillation import KLDivergenceCriterion
from src.losses.ranking import MarginRankingCriterion, RankNetCriterion
from src.losses.regression import (
    ExpectationCriterion,
    HuberCriterion,
    MeanAbsoluteErrorCriterion,
    MeanSquaredErrorCriterion,
    SmoothL1Criterion,
)
from src.losses.segmentation import DiceCriterion, IoUCriterion, TverskyCriterion

__all__ = [
    "ArcFaceCriterion",
    "BinaryCrossEntropyCriterion",
    "CrossEntropyCriterion",
    "DiceCriterion",
    "ExpectationCriterion",
    "FocalCriterion",
    "HuberCriterion",
    "InfoNceCriterion",
    "IoUCriterion",
    "KLDivergenceCriterion",
    "MarginRankingCriterion",
    "MeanAbsoluteErrorCriterion",
    "MeanSquaredErrorCriterion",
    "ProxyAngularCriterion",
    "RankNetCriterion",
    "SigLipCriterion",
    "SmoothL1Criterion",
    "TripletCriterion",
    "TverskyCriterion",
    "WeightedSumCriterion",
    "WrappedCriterion",
]
