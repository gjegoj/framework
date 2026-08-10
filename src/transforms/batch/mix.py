"""Blending and pasting: two samples become one, and so do their labels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

from torchvision.transforms import v2

from src.core.entities import Batch, require_tensor
from src.core.taxonomy import Modality, Objective, Topology
from src.transforms.batch.labels import as_soft, class_counts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.core.entities import DataProfile, Task

PAIRED_WITH = 1
"""How far a sample is from the one it mixes with.

torchvision pairs by rolling the batch by one and documents that as assuming a
shuffled batch, which the training loader is. Labels must roll by the same
amount, or a picture would take one neighbour and its label another.
"""


class LabelMix(ABC):
    """Shared base: one draw mixes the image and every task's label.

    Because these change the image every task shares, they rewrite every task's label
    from the *same* draw — a two-head model must not blend its heads one way and its
    picture another.

    Only global tasks can be served. A blended image has no coherent per-pixel target,
    and metric learning's proxy and margin losses break on soft labels; both are refused
    here rather than discovered after an hour of training.

    The draw and the geometry are torchvision's, reached through the public extension
    points every v2 transform defines: ``make_params`` samples the Beta weight and, for
    CutMix, the box with its clamping and area-adjusted weight; ``transform`` blends or
    pastes. Only the label mixing is ours, because torchvision's assumes one head with
    one class count.

    Parameters:
        tasks (Sequence[Task]): Every task whose label must be rewritten.
        profile (DataProfile): Where the class counts come from.
        alpha (float): Beta parameter; larger values mix more evenly.
        input_name (str): Which input holds the image.
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        profile: DataProfile,
        alpha: float = 1.0,
        input_name: str = Modality.IMAGE,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"{type(self).__name__} needs a positive alpha, got {alpha}.")
        refused = [
            task.name for task in tasks if task.topology is not Topology.GLOBAL or task.objective is Objective.METRIC
        ]
        if refused:
            raise ValueError(
                f"{type(self).__name__} cannot rewrite the targets of {', '.join(refused)}: a mixed "
                f"image has no coherent per-pixel target, and soft labels break metric learning. "
                f"Drop the transform, or the task it cannot serve."
            )
        self._classes = class_counts(tasks, profile)
        self._input_name = input_name
        self._mixer = self._build_mixer(alpha)

    def __call__(self, batch: Batch) -> Batch:
        """Return a new batch; the one given is never written into."""
        image = batch.inputs[self._input_name]
        # ``labels: None`` is what keeps ``transform`` from taking the image for a label;
        # it compares by identity, so any value the image is not will do.
        params: dict[str, Any] = {
            **self._mixer.make_params([image]),
            "labels": None,
            "batch_size": image.shape[0],
        }
        weight = float(params[self._weight_key])
        return Batch(
            inputs={**batch.inputs, self._input_name: self._mixer.transform(image, params)},
            targets={
                **batch.targets,
                **{
                    name: self._mix_label(
                        as_soft(
                            require_tensor(batch.targets[name], task=name, wanted_by="a batch transform"),
                            self._classes[name],
                        ),
                        weight,
                    )
                    for name in self._classes
                },
            },
            meta=batch.meta,
        )

    @staticmethod
    @abstractmethod
    def _build_mixer(alpha: float) -> v2.Transform:
        """The torchvision transform whose draw and geometry this one borrows."""

    @property
    @abstractmethod
    def _weight_key(self) -> str:
        """Which key of ``make_params`` holds the weight the labels should use."""

    @staticmethod
    def _mix_label(label: Tensor, weight: float) -> Tensor:
        return weight * label + (1.0 - weight) * label.roll(PAIRED_WITH, 0)


class MixUp(LabelMix):
    """Blend two images, and their labels, by one sampled weight."""

    @staticmethod
    @override
    def _build_mixer(alpha: float) -> v2.Transform:
        return v2.MixUp(alpha=alpha, num_classes=None)

    @property
    @override
    def _weight_key(self) -> str:
        return "lam"


class CutMix(LabelMix):
    """Paste a rectangle of one image onto another, and weight labels by its area.

    Every pixel comes from exactly one source, which is what distinguishes this
    from blending: the model sees real texture rather than a ghost of two. The
    weight follows the area actually pasted, which clipping at the frame edge
    may have shrunk — torchvision computes that, and it is the piece least worth
    rewriting.
    """

    @staticmethod
    @override
    def _build_mixer(alpha: float) -> v2.Transform:
        return v2.CutMix(alpha=alpha, num_classes=None)

    @property
    @override
    def _weight_key(self) -> str:
        return "lam_adjusted"
