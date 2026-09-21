"""A second network in the run: it teaches, it never learns, and nothing the run leaves behind holds it."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.core import Batch, ModelOutput, Stream, TargetInfo, TensorTree
from src.losses import KullbackLeibler, Loss, MeanSquaredError
from src.losses.build import build_loss
from src.models import Model
from src.tasks import Classification, Regression, Task
from src.training.distillation import DISTILLATION, REPRESENTATION, DistillationLearner

TASK = "species"
CLASSES = {0: "cat", 1: "dog", 2: "bird"}
STUDENT = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
TEACHER = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.5, 0.5]])
SOFT = f"{TASK}/distillation"
STREAM = Stream.POOLED
STUDENT_FEATURES = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
TEACHER_FEATURES = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
ALIGNED = f"{STREAM}/representation"


class Logits(Model):
    """A network answering with a learned tensor, scaled by a learned number.

    Not ``Echo`` from the support module: its answers are plain tensors held in a dict, so moving the
    module leaves them behind — and one of these tests is about exactly that move.

    The scale is not decoration. A forward returning the parameter itself hands a step the weight rather
    than an answer, and `no_grad` stops a graph from being built, not a leaf from already requiring one:
    the teacher would then collect gradients through a test that looked like it was about the learner.
    """

    def __init__(self, logits: Tensor, features: Tensor | None = None) -> None:
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.scale = nn.Parameter(torch.ones(()))
        self.features = None if features is None else nn.Parameter(features.clone())

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(
            outputs={TASK: self.logits * self.scale},
            features={} if self.features is None else {STREAM: self.features * self.scale},
        )


def batch() -> Batch:
    return Batch(inputs={"image": torch.zeros(2, 3, 4, 4)}, targets={TASK: torch.tensor([0, 1])}, count=2)


def taught(
    *,
    student: Tensor = STUDENT,
    teacher: Tensor = TEACHER,
    student_features: Tensor | None = None,
    teacher_features: Tensor | None = None,
    task: Task | None = None,
    **declared: Any,
) -> DistillationLearner:
    """A run of one task over one network, with a second network beside it to agree with.

    Both networks publish no feature stream unless a test says they do, so every run written before
    features could be compared reads here exactly as it read.
    """
    learned = task if task is not None else Classification(TASK, TargetInfo(classes=CLASSES))
    losses: dict[str, Loss] = {learned.name: build_loss(learned.default_loss, learned.facts())}
    return DistillationLearner(
        Logits(student, student_features),
        {learned.name: learned},
        losses,
        teacher=Logits(teacher, teacher_features),
        **declared,
    )


def reported_as[T: Loss](objective: T, column: str) -> T:
    """An objective named the way a builder names the term a run declared: for what that term reads.

    Named there rather than by the learner, because a run writing `log_name` has to keep it — so nothing
    downstream renames an objective it was handed. A test constructing the learner by hand stands where
    the builder would, and does what the builder does.
    """
    objective.log_name = column
    return objective


def pulling(**declared: Any) -> DistillationLearner:
    """A run whose teacher is listened to through a feature stream, which is what these tests are about."""
    return taught(student_features=STUDENT_FEATURES, teacher_features=TEACHER_FEATURES, **declared)


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
    assert [name for name, _ in learner.named_children()] == ["model", "losses", "representation", "loss"]
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


@pytest.mark.parametrize(
    ("declared", "divergence"),
    [
        pytest.param(None, 0.1488416, id="the objective a run gets without declaring one"),
        pytest.param(1.0, 0.1414574, id="unsoftened"),
        pytest.param(4.0, 0.1488416, id="softened as the default is"),
        pytest.param(8.0, 0.1445040, id="softened twice as far"),
    ],
)
def test_the_term_is_the_softened_divergence_over_classes_scaled_by_the_square_of_the_temperature(
    declared: float | None, divergence: float
) -> None:
    """The arithmetic itself, so that moving it elsewhere is observed rather than assumed.

    Every other test here says what the term *does* — zero where the two answer alike, positive where
    they do not, weighed the same at any temperature — and all of them would go on passing if the
    formula were replaced by a different one of the same shape. These numbers were worked out from the
    definition apart from the code under test: soften both rows by the temperature, sum the divergence
    over the classes, average over the batch, multiply by the temperature squared. They agree with
    ``kl_div`` to six places, which is float32 against the arithmetic done in float64.
    """
    objective = None if declared is None else reported_as(KullbackLeibler(temperature=declared), DISTILLATION)

    assert float(terms(taught(loss=objective))[SOFT].detach()) == pytest.approx(divergence, rel=1e-5)


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
        learner = taught(loss=reported_as(KullbackLeibler(temperature=temperature), DISTILLATION))
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


def test_how_far_the_features_are_from_the_teachers_is_reported_under_the_stream_that_was_read() -> None:
    """One term per stream, prefixed by the stream, as a task's term is prefixed by the task."""
    assert ALIGNED in terms(pulling(representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)}))


def test_a_teacher_whose_features_the_student_already_matches_adds_nothing() -> None:
    """This term is a distance, so two identical representations are zero of it; a run reporting
    otherwise would be descending something that is not the distance its name claims."""
    learner = taught(
        student_features=TEACHER_FEATURES,
        teacher_features=TEACHER_FEATURES,
        representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)},
    )

    assert terms(learner)[ALIGNED].item() == pytest.approx(0.0)


def test_the_share_the_teacher_is_worth_scales_the_features_term_as_it_scales_the_answers() -> None:
    """One number for the whole of what is learned from a second network, both halves of it."""
    reported = terms(pulling(representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)}, weight=3.0))

    assert reported[f"{ALIGNED}/contribution"].item() == pytest.approx(reported[ALIGNED].item() * 3.0)


def test_a_run_pulling_features_alone_descends_no_divergence_nobody_wrote() -> None:
    """The default over the answers is what an algorithm makes when a run declared nothing at all; a run
    that did declare its terms gets those and no second one added silently beside them."""
    reported = terms(pulling(representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)}))

    assert ALIGNED in reported
    assert SOFT not in reported


def test_a_term_that_named_itself_keeps_that_name_through_the_learner() -> None:
    """`log_name` is how a run tells two terms over one reading apart, so a learner naming them all alike
    would undo exactly the distinction the run wrote down."""
    named = KullbackLeibler()
    named.log_name = "divergence"

    assert f"{TASK}/divergence" in terms(taught(loss=named))


def test_a_teacher_whose_features_are_another_shape_is_refused_rather_than_broadcast() -> None:
    """`mse_loss` broadcasts, so a teacher of one number against a student of two gives a finite number
    and a total that reads like distillation."""
    learner = taught(
        student_features=STUDENT_FEATURES,
        teacher_features=torch.ones(2, 1),
        representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)},
    )

    with pytest.raises(ValueError, match=STREAM):
        terms(learner)


def test_a_network_publishing_no_such_stream_is_refused_by_the_name_of_the_stream() -> None:
    """A teacher arriving whole by `_target_` publishes whatever it publishes, and only its answer says
    what that is — the same reason the refusal about answers stands where this one does."""
    learner = taught(
        student_features=STUDENT_FEATURES,
        representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)},
    )

    with pytest.raises(ValueError, match=STREAM):
        terms(learner)


def test_a_task_that_is_not_a_distribution_over_classes_is_distilled_through_its_features() -> None:
    """Softening spreads confidence over classes and a regression has none — but a representation is a
    representation, and a run learning only that from its teacher has nothing to soften."""
    learner = pulling(
        task=Regression(TASK, TargetInfo()),
        student=torch.tensor([[1.0], [2.0]]),
        teacher=torch.tensor([[1.0], [2.0]]),
        representation={STREAM: reported_as(MeanSquaredError(), REPRESENTATION)},
    )

    assert ALIGNED in terms(learner)
