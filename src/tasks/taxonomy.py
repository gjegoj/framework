"""Task taxonomy: the two internal axes a task is composed from.

Users pick a familiar preset (``classification``, ``segmentation``, ...); under
the hood each preset is a point in the ``Topology x Objective`` grid.
"""

from enum import StrEnum


class Topology(StrEnum):
    """Structural shape of the feature→head path: which head, which feature stream,
    and how N items reach the head. New topologies (detection/OCR) are added here
    only together with their registered strategy — not declared speculatively.
    """

    GLOBAL = "global"  # one prediction per sample (classification / regression / arcface)
    DENSE = "dense"  # one prediction per pixel (segmentation)
    RANKING = "ranking"  # N views through ONE shared backbone → [B, N, D] (Siamese / triplet)
    MULTISTREAM = "multistream"  # N streams from N SEPARATE encoders → [B, N, D] (CLIP / SIGLIP)


class Objective(StrEnum):
    """Label semantics (which codec + criterion + activation + metric mode)."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTINUOUS = "continuous"
    METRIC = "metric"  # implicit target: supervision from pair/triplet/batch structure
