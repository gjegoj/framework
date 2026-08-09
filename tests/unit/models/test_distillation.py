"""``DistilledModel``: a student judged additionally against a frozen teacher's logits."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from src.core import Batch, Loss, Model, Prediction, StepResult
from src.models import DistilledModel, without_teachers
from tests.support.narrowing import tensor


class Student(Model):
    """Scales its input into 'label' logits and reports a fixed hard loss."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def step(self, batch: Batch) -> StepResult:
        return StepResult(
            loss=Loss.part("ce", torch.tensor(2.0)),
            prediction=self.predict(batch),
            targets={"label": batch.targets["label"]},
        )

    def predict(self, batch: Batch) -> Prediction:
        logits = batch.inputs["image"] * self.scale
        return Prediction(outputs={"label": torch.softmax(logits, dim=1)}, logits={"label": logits})

    def task_parameters(self, task_name: str) -> list[nn.Parameter]:
        return [self.scale]


class Teacher(Model):
    """Returns a constant logit tensor, and records how often it was asked for one."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))
        self.asked = 0

    def step(self, batch: Batch) -> StepResult:
        return StepResult(loss=Loss.part("ce", torch.tensor(0.0)), prediction=self.predict(batch), targets={})

    def predict(self, batch: Batch) -> Prediction:
        self.asked += 1
        logits = torch.full_like(batch.inputs["image"], float(self.value))
        return Prediction(outputs={"label": torch.softmax(logits, dim=1)}, logits={"label": logits})


class Silent(Model):
    """A family whose native output has no pre-activation form — boxes, say."""

    def step(self, batch: Batch) -> StepResult:
        return StepResult(loss=Loss.part("ce", torch.tensor(1.0)), prediction=self.predict(batch), targets={})

    def predict(self, batch: Batch) -> Prediction:
        return Prediction(outputs={"label": torch.zeros(2, 4)})


class Constant(nn.Module):
    """A criterion returning a fixed soft loss, so the arithmetic is checkable by hand."""

    def __init__(self, value: float = 3.0) -> None:
        super().__init__()
        self.value = value

    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        return Loss.part("kl", torch.tensor(self.value))


def batch() -> Batch:
    return Batch(inputs={"image": torch.ones(2, 4)}, targets={"label": torch.zeros(2, dtype=torch.long)})


def distilled(teachers: int = 1, student: Model | None = None, soft: float = 3.0) -> DistilledModel:
    return DistilledModel(
        student=student or Student(),
        teachers=[Teacher(float(index)) for index in range(teachers)],
        criterion=Constant(soft),  # type: ignore[arg-type]
    )


def test_the_training_loss_is_the_hard_one_plus_the_soft_one() -> None:
    """Nothing here scales anything: how strongly a criterion pulls is the criterion's own weight."""
    model = distilled(soft=3.0)
    model.train()

    total = model.step(batch()).loss.total

    assert total.item() == pytest.approx(2.0 + 3.0)


def test_both_losses_are_reported_under_their_own_names() -> None:
    """A run has to see which signal moved; one merged number hides the trade between them."""
    model = distilled()
    model.train()

    parts = model.step(batch()).loss.parts

    assert "ce" in parts
    assert any("kl" in name for name in parts)


def test_off_training_the_teachers_are_not_even_asked() -> None:
    """Validation has to report the same quantity an undistilled run does, and pay nothing for the teacher."""
    model = distilled()
    model.eval()

    result = model.step(batch())

    assert result.loss.total.item() == pytest.approx(2.0)
    assert all(teacher.asked == 0 for teacher in model.teachers)


def test_several_teachers_are_averaged_and_one_teacher_is_that_teacher() -> None:
    """Their logits are one soft target; averaging is what makes an ensemble of them."""
    ensemble = distilled(teachers=3)
    ensemble.train()
    alone = distilled(teachers=1)
    alone.train()

    ensemble.step(batch())

    assert all(teacher.asked == 1 for teacher in ensemble.teachers)
    assert torch.allclose(ensemble._guidance(batch())["label"], torch.full((2, 4), 1.0))
    assert torch.allclose(alone._guidance(batch())["label"], torch.zeros(2, 4))


def test_prediction_and_task_parameters_are_the_students() -> None:
    """The teacher shapes the loss and nothing else; inference must not know it existed."""
    model = distilled()
    assert isinstance(model.student, Student)

    assert torch.allclose(
        tensor(model.predict(batch()).outputs["label"]), tensor(model.student.predict(batch()).outputs["label"])
    )
    assert list(model.task_parameters("label")) == [model.student.scale]


def test_the_teachers_stay_out_of_the_state_dict() -> None:
    """They are frozen scaffolding: every checkpoint and the EMA copy would otherwise carry them."""
    model = distilled()

    assert not any("teacher" in name for name in model.state_dict())


def test_the_teachers_never_learn() -> None:
    """A teacher that drifted would move the target the student is chasing."""
    model = distilled()

    assert all(not parameter.requires_grad for teacher in model.teachers for parameter in teacher.parameters())
    assert all(not teacher.training for teacher in model.teachers)


def test_a_model_family_without_logits_is_refused_by_name() -> None:
    """Silently distilling nothing would look exactly like distillation that did not help."""
    model = distilled(student=Silent())
    model.train()

    with pytest.raises(ValueError, match="Silent"):
        model.step(batch())


def test_without_teachers_returns_the_student_and_leaves_anything_else_alone() -> None:
    """Measured: a traced graph carries an unused teacher, so the artifact is written from the student."""
    model = distilled()
    plain = Student()

    assert without_teachers(model) is model.student
    assert without_teachers(plain) is plain


def test_the_student_sits_under_the_name_the_checkpoint_reader_looks_for() -> None:
    """The reader strips exactly one prefix by name; renaming the attribute would silently strand it."""
    model = distilled()

    assert getattr(model, DistilledModel.STUDENT) is model.student
