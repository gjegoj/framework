"""Task taxonomy: the two internal axes a task is composed from.

Users pick a familiar preset (``classification``, ``segmentation``, ...); under
the hood each preset is a point in the ``Topology x Objective`` grid.
"""

from enum import StrEnum


class Topology(StrEnum):
    """Spatial structure of the prediction (which head + feature stream)."""

    GLOBAL = "global"  # one prediction per sample (classification/regression)
    DENSE = "dense"  # one prediction per pixel (segmentation)
    SET = "set"  # one prediction per object/region (detection)
    SEQUENCE = "sequence"  # one prediction per token (OCR)
    EMBEDDING = "embedding"  # a vector (retrieval / contrastive)


class Objective(StrEnum):
    """Label semantics (which codec + criterion + activation + metric mode)."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTINUOUS = "continuous"
