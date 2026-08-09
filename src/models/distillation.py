"""A student trained beside frozen teachers, judged additionally against their logits."""

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

    A decorator over ``Model`` rather than a second training module, because
    distillation adds one term to a step and changes nothing else: ``step`` returns the
    student's step with the soft term added, and the prediction, the per-task
    parameters and inference stay the student's own. Off training the teachers do not
    run at all, which keeps a validation loss comparable with an undistilled run's.

    The soft term is an ordinary loss term — scoped to its task, logged under its own
    name beside the hard one, carrying its own weight — so nothing here multiplies
    anything.

    The teachers are held **outside** the module tree on purpose: registered, a frozen
    model would be written into every checkpoint and copied again by the EMA callback,
    which averages the whole module. The price is that Lightning never moves them, paid
    per step by aligning them to the student's own device and dtype, read fresh each
    time so nothing drifts between what a hook assumed and what the trainer did.

    Parameters:
        student (Model): The model being trained; everything but the loss is its own.
        teachers (Sequence[Model]): Frozen soft-target providers. Their raw logits
            are averaged, so one teacher is a one-element sequence rather than a
            case of its own.
        criterion (Criterion): How student and teacher logits are compared —
            ``kl_divergence`` and its temperature, or anything else with the port.
            How strongly the comparison pulls is the criterion's own weight, the
            way every other loss term in this framework carries its own.
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
        # A plain list, not a ModuleList: registered, these frozen models would ride
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
    """What a run is about once training is over: the student, without its scaffolding.

    A checkpoint carries the shipped model's own keys, and those do not load into
    the decorator — so restoring the epoch a run kept has to go through this.
    Export goes through it too, though the teachers are already invisible there
    (measured: a traced graph carries what is registered, and they are held in a
    plain list); the call says so where a reader looks, and keeps saying it if
    that ever changes.

    Returns anything that is not distilled unchanged, so no caller branches.
    """
    return model.student if isinstance(model, DistilledModel) else model
