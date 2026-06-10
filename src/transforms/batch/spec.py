"""TargetSpec: what a batch transform needs to know about one task's target.

Built in the composition root (it knows each task's topology and class count) and
injected into a batch transform so the transform can rewrite *every* target the
shared-image change affects — coherently per topology.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.tasks.taxonomy import Topology


@dataclass(frozen=True)
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
