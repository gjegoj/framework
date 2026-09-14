"""A task with no column: what it is learned against is which sample of the batch each draw came from."""

from __future__ import annotations

from typing import ClassVar, override

import torch
from torch import Tensor
from torch.nn.functional import normalize

from src.core import FEATURE_AXIS, Axis, Batch, Representation, TargetInfo, TensorTree
from src.tasks.base import LossDeclaration, Task
from src.tasks.registry import task_registry


@task_registry.register("contrastive")
class Contrastive(Task):
    """Draws of one picture are learned to point the same way, and draws of different pictures apart.

    The kind for a folder of pictures nobody labelled. What stands in for a label is the drawing itself:
    two views of one sample are the only pair in the batch that belong together, so the supervision is
    which row a sample is — derived from the batch rather than read from a column, which is why this is
    the one kind whose declaration names no ``target_column``.

    What it answers with is a direction, as metric learning does, and for the same reason: the length of
    an embedding carries nothing and the angle carries everything. The width of that direction is a
    choice of the model rather than something the data settles, so it is declared on the kind.

    No readings by default. A vocabulary is what the metrics of this framework score against and this
    kind has none; what a contrastive run is actually judged by is a second run that uses its encoder,
    which is a different declaration and not a number this one can report.

    Parameters:
        embedding_dim: How wide the direction this task answers with is.
    """

    output_axis: ClassVar[str] = Axis.EMBEDDING
    publishes: ClassVar[Representation] = Representation.DIRECTION

    def __init__(
        self, name: str, info: TargetInfo, *, embedding_dim: int, weight: float = 1.0, lr: float | None = None
    ) -> None:
        if embedding_dim < 1:
            raise ValueError(
                f"'embedding_dim' is how wide the direction this task answers with is, and there is no "
                f"defensible default for it; it was declared as {embedding_dim}."
            )
        super().__init__(name, info, weight=weight, lr=lr)
        self.embedding_dim = embedding_dim

    def out_features(self) -> int:
        return self.embedding_dim

    @property
    def default_loss(self) -> LossDeclaration:
        return "info_nce"

    def loss_target(self, batch: Batch) -> Tensor:
        """Which sample of the batch a draw came from, which is the only answer this kind has.

        Built here rather than read, because there is no column to read: the batch *is* the supervision,
        and a run that had to write a row number into its table would be declaring what the loader
        already knows.
        """
        return torch.arange(batch.count, device=batch.device)

    def metric_view(self, batch: Batch) -> Tensor:
        """The same answer, for a reading declared by a run that has one to declare."""
        return self.loss_target(batch)

    @override
    def publish(self, projected: Tensor) -> TensorTree:
        """A unit vector: only the direction carries what was learned, and this is what the artifact emits."""
        return normalize(projected, dim=FEATURE_AXIS)
