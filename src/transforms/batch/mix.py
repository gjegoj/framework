"""Blending and pasting: two samples become one, and so do their labels."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, override

from torchvision.transforms import v2

from src.core.entities import Batch, require_tensor
from src.core.taxonomy import Modality, OutputTopology
from src.transforms.batch.ports import refuse_unservable, unbound

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from src.tasks import Task

PAIRED_WITH = 1
"""How far a sample is from the one it mixes with.

torchvision pairs by rolling the batch by one and documents that as assuming a
shuffled batch, which the training loader is. Labels must roll by the same
amount, or a picture would take one neighbour and its label another.
"""


class LabelMix(ABC):
    """Shared base: one draw mixes the image and every task's label.

    Every task's label is rewritten from the *same* draw, because the image every task
    shares changed. Only global tasks are served: a blended image has no coherent per-pixel
    target, and metric learning's losses break on soft labels — refused when the tasks are
    bound (``for_tasks``), before the first batch, not an hour in. The draw and geometry
    are torchvision's (``make_params``, ``transform``); only the label mixing is ours,
    because torchvision's assumes one head.

    Parameters:
        alpha (float): Beta parameter; larger values mix more evenly.
        input_name (str): Which input holds the image.
    """

    def __init__(self, alpha: float = 1.0, input_name: str = Modality.IMAGE) -> None:
        if alpha <= 0:
            raise ValueError(f"{type(self).__name__} needs a positive alpha, got {alpha}.")
        self._tasks: dict[str, Task] | None = None
        self._input_name = input_name
        self._mixer = self._build_mixer(alpha)

    def for_tasks(self, tasks: Sequence[Task]) -> Callable[[Batch], Batch]:
        """A copy bound to these tasks — the ``BatchTransform`` port.

        A copy rather than a rebinding in place, so the declared transform stays what
        config said and the unbound refusal below stays honest for it. Each task knows how
        its own target softens and carries the facts that takes.
        """
        refuse_unservable(
            self,
            tasks,
            {OutputTopology.GLOBAL},
            "a mixed image has no coherent per-pixel target, and soft labels break metric learning.",
        )
        bound = copy.copy(self)
        bound._tasks = {task.name: task for task in tasks}
        return bound

    def __call__(self, batch: Batch) -> Batch:
        """Return a new batch; the one given is never written into."""
        if self._tasks is None:
            raise unbound(self)
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
                        task.kind.soften(
                            require_tensor(batch.targets[name], task=name, wanted_by="a batch transform"), task.facts
                        ),
                        weight,
                    )
                    for name, task in self._tasks.items()
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
