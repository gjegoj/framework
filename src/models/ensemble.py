"""Model ensembling: combine several built models into one averaged-logit model.

Where ``assembly.py`` composes a single backbone + heads into a ``CompositeModel``,
this composes N already-built models into one. ``TeacherEnsemble`` is the frozen,
logit-averaging ensemble used as the soft-target provider for knowledge distillation
(consumed by ``DistillationLitModule``); it is a pure ``nn.Module`` with no training
or Lightning dependency, so it lives in the model layer, not the training layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class TeacherEnsemble(nn.Module):
    """Frozen teacher models producing averaged soft targets per task.

    Raw logits are averaged (mean over teachers), matching the established
    practice; probability averaging can become an option later. The forward runs
    under ``torch.no_grad()`` (not ``inference_mode`` — the result is consumed as
    a constant inside the student's autograd graph).

    Parameters:
        teachers (Sequence[nn.Module]): Models mapping ``inputs`` to a
            ``ModelOutput``; typically ``CompositeModel``s with loaded weights.
    """

    def __init__(self, teachers: Sequence[nn.Module]) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("TeacherEnsemble needs at least one teacher.")
        self._teachers = nn.ModuleList(teachers)
        self._teachers.requires_grad_(False)
        self.eval()

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        with torch.no_grad():
            outputs = [teacher(inputs) for teacher in self._teachers]
        if len(outputs) == 1:
            return dict(outputs[0].task_logits)  # single teacher: its logits ARE the average
        # sum/n instead of stack().mean(): avoids materializing an [N, *shape] copy per task.
        return {
            name: sum((output.task_logits[name] for output in outputs[1:]), start=first) / len(outputs)
            for name, first in outputs[0].task_logits.items()
        }

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        # Typed delegate to nn.Module.__call__ (see core.ports docstring); preserves hooks.
        return cast("dict[str, Tensor]", super().__call__(inputs))
