"""Task taxonomy: the two internal axes a task is composed from.

Users pick a familiar preset (``classification``, ``segmentation``, ...); under
the hood each preset is a point in the ``Topology x Objective`` grid.

These enums are the domain vocabulary a ``Task`` is typed by (``core/entities.py``),
so they live in the center: the task use-case layer, the batch transforms, the
visualization annotators and the export guard all consume them, and none of those
outer layers should own a word the core entity references.
"""

from __future__ import annotations

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


EXPORTABLE_TOPOLOGIES: frozenset[Topology] = frozenset({Topology.GLOBAL, Topology.DENSE})


class Objective(StrEnum):
    """Label semantics (which adapter + criterion + activation + metric mode)."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTINUOUS = "continuous"
    METRIC = "metric"  # implicit target: supervision from pair/triplet/batch structure
