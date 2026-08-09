"""Distillation criteria: a student judged against a teacher's outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn
from torch.nn import functional

from src.losses.base import WrappedCriterion
from src.losses.registry import criterion_registry

if TYPE_CHECKING:
    from torch import Tensor


class KLDivergenceLoss(nn.Module):
    """Temperature-scaled KL divergence from teacher logits to student logits.

    ``KL(softmax(teacher/T) || softmax(student/T)) * T²`` — the ``T²`` factor
    keeps soft-target gradients on the scale of hard-target ones, which is what
    lets the two be added with an honest weight. The class dimension is dim 1
    and everything after it rides along, so one module serves ``[B, C]`` and
    dense ``[B, C, H, W]`` alike. The teacher side is detached: gradients reach
    the student argument only.

    ``temperature`` is a plain number read anew each step, so the ``anneal``
    callback can cool it over the run.

    Parameters:
        temperature (float): Softening; 1.0 is plain KL, higher exposes more of
            the teacher's dark knowledge in the small logits.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"KLDivergenceLoss temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        if target.shape != logits.shape or not target.is_floating_point():
            raise ValueError(
                f"Distillation compares logits with logits: the student produced "
                f"{tuple(logits.shape)} but the target is {tuple(target.shape)} "
                f"{target.dtype}. Its target is a teacher's output, not a class label."
            )
        student = functional.log_softmax(logits / self.temperature, dim=1)
        teacher = functional.softmax(target.detach() / self.temperature, dim=1)
        divergence = functional.kl_div(student, teacher, reduction="none")
        return divergence.sum(dim=1).mean() * (self.temperature * self.temperature)


@criterion_registry.register("kl_divergence")
class KLDivergenceCriterion(WrappedCriterion):
    """Distillation KL as a criterion.

    Parameters:
        **kwargs: Forwarded verbatim to :class:`KLDivergenceLoss`
            (``temperature``).
    """

    part_name: ClassVar[str] = "kl"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(KLDivergenceLoss(**kwargs))
