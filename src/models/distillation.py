"""A student trained beside frozen teachers, with an extra loss against their logits."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import torch
from torch import nn

from src.core.entities import Loss, StepResult
from src.core.ports import Model

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch import Tensor

    from src.core.entities import Batch, Prediction
    from src.core.ports import Criterion


class DistilledModel(Model):
    """The student, plus a soft loss against the averaged logits of frozen teachers.

    A decorator over ``Model``: ``step`` returns the student's step with the soft term added
    (an ordinary loss term, scoped to its task, logged under its own name, carrying its own
    weight); prediction, per-task parameters and inference stay the student's. Off training
    the teachers do not run. The teachers are held *outside* the module tree — registered,
    they would be written into every checkpoint and averaged by the EMA callback — so they
    are aligned to the student's device and dtype on every step.

    Parameters:
        student (Model): The model being trained; everything but the loss is its own.
        teachers (Sequence[Model]): Frozen soft-target providers; their raw logits are averaged.
        criterion (Criterion): How student and teacher logits are compared —
            ``kl_divergence`` and its temperature; its weight is the criterion's own.
    """

    STUDENT: ClassVar[str] = "student"
    """The attribute holding the student, and so the one key path a run's checkpoint gains.

    Published rather than copied: the checkpoint reader strips this prefix to make
    one file load into a distilled model and a plain one alike, and the backbone's
    dot-path is composed with it. A rename only those readers knew about would
    strand them both.
    """

    def __init__(self, student: Model, teachers: Sequence[Model], criterion: Criterion) -> None:
        super().__init__()
        if not teachers:
            raise ValueError("DistilledModel needs at least one teacher; without one there is nothing to distil from.")
        self.student = student
        # A plain list, not a ModuleList: registered, these frozen models would be written
        # into every checkpoint and into the EMA callback's copy of the module.
        self.teachers = list(teachers)
        for teacher in self.teachers:
            teacher.eval()
            teacher.requires_grad_(False)
        self.criterion = criterion

    @override
    def step(self, batch: Batch) -> StepResult:
        result = self.student.step(batch)
        if not self.training:
            return result
        learned = _logits_of(self.student, result.prediction)
        guidance = self._guidance(batch)
        shared = [name for name in learned if name in guidance]
        if not shared:
            raise ValueError(
                f"The student and the teachers share no task to distil: the student reports "
                f"{', '.join(sorted(learned))} and the teachers {', '.join(sorted(guidance))}."
            )
        soft = Loss.sum([self.criterion(learned[name], guidance[name]).scoped(name) for name in shared])
        return StepResult(loss=result.loss + soft, prediction=result.prediction, targets=result.targets)

    @override
    def predict(self, batch: Batch) -> Prediction:
        return self.student.predict(batch)

    @override
    def task_parameters(self, task_name: str) -> Iterable[nn.Parameter]:
        return self.student.task_parameters(task_name)

    @override
    def criterion_of(self, task_name: str) -> nn.Module | None:
        """The student's: the teachers are frozen, and the soft term is not a task's own."""
        return self.student.criterion_of(task_name)

    @property
    @override
    def architecture(self) -> str:
        """The student's: a run is filed under what learned, not under the scaffolding around it."""
        return self.student.architecture

    def _guidance(self, batch: Batch) -> dict[str, Tensor]:
        """The teachers' averaged logits, on the student's own device and dtype."""
        self._align_to_student()
        with torch.no_grad():
            answers = [_logits_of(teacher, teacher.predict(batch)) for teacher in self.teachers]
        return {name: torch.stack([answer[name] for answer in answers]).mean(dim=0) for name in answers[0]}

    def _align_to_student(self) -> None:
        """Put the teachers where the student is, reading that afresh each step.

        They are off the module tree, so Lightning never moves them. Doing it here
        rather than in a lifecycle hook means the alignment cannot go stale: it sees
        the device and dtype the student actually has now, whatever moved it there.
        """
        reference = next(self.student.parameters(), None)
        if reference is None:
            return
        for teacher in self.teachers:
            held = next(teacher.parameters(), None)
            if held is not None and (held.device != reference.device or held.dtype != reference.dtype):
                teacher.to(device=reference.device, dtype=reference.dtype)


def _logits_of(model: Model, prediction: Prediction) -> dict[str, Tensor]:
    """One model's pre-activation values, refusing a family that has none to give."""
    if prediction.logits is None:
        raise ValueError(
            f"{type(model).__name__} reports no logits, and distillation compares logits with logits — "
            "a temperature cannot be applied to what an activation has already collapsed. A model family "
            "taking part in distillation has to fill Prediction.logits."
        )
    return prediction.logits


def without_teachers(model: Model) -> Model:
    """The student without its scaffolding — what a run is about once training is over.

    A checkpoint carries the student's own keys, which do not load into the decorator, so
    restoring and exporting go through this. Returns anything not distilled unchanged.
    """
    return model.student if isinstance(model, DistilledModel) else model
