"""Batch-transform contracts: the ``BatchTransform`` port and the ``TargetSpec`` it consumes.

A batch transform mixes/stitches whole samples in a collated ``Batch`` (MixUp,
CutMix, Mosaic). Because it changes the shared image, it must rewrite every task's
target coherently — so it is injected the tasks' ``TargetSpec`` list, built in the
composition root (which knows each task's topology and class count).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tasks.taxonomy import Topology

if TYPE_CHECKING:
    from src.core.entities import Batch


class BatchTransform(ABC):
    """A cross-sample transform applied to a collated training ``Batch``.

    Unlike a per-sample (Albumentations) transform that runs in the data layer, a
    batch transform mixes/combines whole samples (MixUp, CutMix, Mosaic) and so
    needs the collated batch. Because it changes the *shared* image, it must
    rewrite **every** task's target coherently; concrete transforms are injected
    with the tasks' ``TargetSpec`` list and declare a class attribute
    ``supported_topologies: frozenset[Topology]`` (the topologies whose target
    they can re-derive). The composition root guards incompatible combinations
    (e.g. MixUp + a DENSE head) at build time. Label-mixing transforms return soft
    targets; the task adapter turns those into a ``TargetView`` (soft for loss,
    hard for metrics).
    """

    supported_topologies: frozenset[Topology] = frozenset()

    @abstractmethod
    def __call__(self, batch: Batch) -> Batch:
        """Return the transformed batch (a new ``Batch``; inputs not mutated)."""


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Describes one task's target for a batch transform.

    Parameters:
        key (str): The ``Batch.targets`` key (task name).
        topology (Topology): Output structure — decides how the target is rewritten
            (GLOBAL → label-style, DENSE → mask-style).
        num_classes (int): Class count, needed to one-hot a GLOBAL label before mixing.
    """

    key: str
    topology: Topology
    num_classes: int
