"""A second network in the run: it teaches, it never learns, and nothing the run leaves behind holds it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Batch, ModelOutput, TargetInfo, TensorTree
from src.losses import Loss
from src.losses.build import build_loss
from src.models import Model
from src.tasks import Classification, Regression, Task
from src.training.distillation import DistillationLearner

TASK = "species"
CLASSES = {0: "cat", 1: "dog", 2: "bird"}
STUDENT = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
TEACHER = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.5, 0.5]])
SOFT = f"{TASK}/distillation"


class Logits(Model):
    """A network answering with a learned tensor, scaled by a learned number.

    Not ``Echo`` from the support module: its answers are plain tensors held in a dict, so moving the
    module leaves them behind — and one of these tests is about exactly that move.

    The scale is not decoration. A forward returning the parameter itself hands a step the weight rather
    than an answer, and `no_grad` stops a graph from being built, not a leaf from already requiring one:
    the teacher would then collect gradients through a test that looked like it was about the learner.
    """

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(outputs={TASK: self.logits * self.scale})


def batch() -> Batch:
    return Batch(inputs={"image": torch.zeros(2, 3, 4, 4)}, targets={TASK: torch.tensor([0, 1])}, count=2)


def taught(
    *, student: Tensor = STUDENT, teacher: Tensor = TEACHER, task: Task | None = None, **declared: Any
) -> DistillationLearner:
    """A run of one task over one network, with a second network beside it to agree with."""
    learned = task if task is not None else Classification(TASK, TargetInfo(classes=CLASSES))
    losses: dict[str, Loss] = {learned.name: build_loss(learned.default_loss, learned.facts())}
    return DistillationLearner(Logits(student), {learned.name: learned}, losses, teacher=Logits(teacher), **declared)


def student_of(learner: DistillationLearner) -> Logits:
    """The network being taught, as what it is, so the one parameter it learns can be read back."""
    student = learner.model
    assert isinstance(student, Logits)
    return student


def terms(learner: DistillationLearner) -> Mapping[str, Tensor]:
    output = learner.step(batch())
    assert output.loss is not None
    return output.loss.breakdown()


def test_the_teacher_is_nowhere_the_run_writes_itself_down() -> None:
    """A checkpoint, a summary, an export and the optimizer all read the module tree, and it is not in it."""
    learner = taught()

    assert not any("teacher" in name for name in learner.state_dict())
    assert [name for name, _ in learner.named_children()] == ["model", "losses"]
    grouped = {id(parameter) for group in learner.parameter_groups() for parameter in group["params"]}
    assert grouped.isdisjoint({id(parameter) for parameter in learner.teacher.parameters()})


def test_the_teacher_never_learns_and_never_leaves_evaluation() -> None:
    """It is a fixed opinion to move towards; a teacher that drifted would be a second student."""
    learner = taught()
    learner.train()

    output = learner.step(batch())
    assert output.loss is not None
    output.loss.total.backward()

    assert learner.teacher.training is False
    assert all(parameter.grad is None for parameter in learner.teacher.parameters())
    assert student_of(learner).logits.grad is not None, "the student learned nothing, so this proves nothing"


def test_how_far_the_student_is_from_the_teacher_is_reported_under_its_own_name() -> None:
    """A report shows what the run descends; a term folded silently into the total is not readable."""
    assert SOFT in terms(taught())


def test_a_teacher_the_student_already_agrees_with_adds_nothing() -> None:
    """The term is a divergence: it is zero exactly when the two answer alike, and positive otherwise."""
    agreed = terms(taught(teacher=STUDENT))
    apart = terms(taught())

    assert float(agreed[SOFT].detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(apart[SOFT].detach()) > 0.0


def test_the_teacher_is_listened_to_in_evaluation_as_well() -> None:
    """`val/loss` and `train/loss` have to sum the same terms, or the number a run is kept by is another one."""
    learner = taught()
    learner.eval()

    assert SOFT in terms(learner)


def test_the_temperature_leaves_the_weight_meaning_what_it_meant() -> None:
    """Softening the two distributions shrinks the term; without the scale, `weight` would mean less at every step.

    Measured on random logits: the gradient falls fourfold per doubling of the temperature, so a run that
    raised it would have to raise `weight` by the same factor to descend the same objective.
    """
    gradients = []
    for temperature in (1.0, 8.0):
        learner = taught(temperature=temperature)
        output = learner.step(batch())
        assert output.loss is not None
        output.loss.breakdown()[SOFT].backward()
        learned = student_of(learner).logits.grad
        assert learned is not None
        gradients.append(float(learned.abs().mean()))

    assert gradients[1] == pytest.approx(gradients[0], rel=0.2)


def test_a_task_whose_answer_is_not_a_distribution_over_classes_is_refused() -> None:
    """Softening needs classes to spread the probability over; a number has none, and a silent skip is worse."""
    with pytest.raises(ValueError) as refusal:
        taught(task=Regression("age", TargetInfo()))

    assert "age" in str(refusal.value)


def test_a_teacher_answering_in_another_space_is_refused_rather_than_broadcast() -> None:
    """Two distributions of different widths are not far apart or close: they are not comparable at all.

    `kl_div` broadcasts, so a teacher answering over one class against a student over three produced a
    finite 53.72823 and a total that read like distillation. Whether the two were ever asked the same
    question is exactly what the number cannot say, and the builder can only settle it for a teacher it
    composed itself — one arriving whole by `_target_` is checked here, where both answers are in hand.
    """
    learner = taught(teacher=torch.tensor([[1.0], [0.5]]))

    with pytest.raises(ValueError, match=r"answers \[2, 1\] — over 1 class"):
        learner.step(batch())


def test_the_teacher_is_moved_wherever_the_run_moved_everything_else() -> None:
    """Held outside the module tree, it is not carried along by the move that puts the run on its device."""
    learner = taught()
    learner.to("meta")

    output = learner.step(Batch(inputs={}, targets={TASK: torch.tensor([0, 1], device="meta")}, count=2))

    assert output.loss is not None and output.loss.total.device.type == "meta"


def test_the_share_of_the_total_a_weighted_teacher_is_worth_is_reported_too() -> None:
    """A term compares runs whatever weight each gave it; a share says which part of the total it is.

    Reported without one, the only distillation row a reader finds is the number the objective did not
    descend — every task term explains itself this way, and the teacher's is a term beside them.
    """
    reported = terms(taught(weight=0.5))

    assert f"{SOFT}/contribution" in reported
    assert float(reported[f"{SOFT}/contribution"].detach()) == pytest.approx(float(reported[SOFT].detach()) * 0.5)
