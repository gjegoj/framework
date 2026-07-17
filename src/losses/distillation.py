"""Distillation criteria: soft-target losses against teacher outputs.

The target of a distillation criterion is not a data-layer label but another
model's logits — still just a tensor, so the standard ``Criterion`` port
signature applies unchanged.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor

from src.core.entities import LossResult
from src.core.ports import Criterion
from src.losses.registry import criteria


@criteria.register("kl_divergence")
class KLDivergenceCriterion(Criterion):
    """Temperature-scaled KL divergence between student logits and teacher logits.

    Computes ``KL(softmax(teacher/T) || softmax(student/T)) * T^2`` — the ``T^2``
    factor keeps soft-target gradients on the same scale as hard-target ones.
    Softmax runs over ``dim=1``, so one instance serves GLOBAL ``[B, C]`` and
    DENSE ``[B, C, H, W]`` alike. The teacher side is detached: gradients flow
    to the student argument only.

    Parameters:
        temperature (float): Softening temperature (``> 0``). ``1.0`` -> plain KL.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature

    def forward(self, logits: Tensor, target: Tensor) -> LossResult:
        student_log_probabilities = F.log_softmax(logits / self.temperature, dim=1)
        teacher_probabilities = F.softmax(target.detach() / self.temperature, dim=1)
        divergence = F.kl_div(student_log_probabilities, teacher_probabilities, reduction="none")
        value = divergence.sum(dim=1).mean() * (self.temperature * self.temperature)
        return LossResult(total=value, components={"kl": value})
