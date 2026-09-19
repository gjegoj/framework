"""Learning from a second network as well as from the data: what it answers is the other half of the objective."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast, override

import torch
from torch import Tensor

from src.core import FEATURE_AXIS, Batch, LossOutput, ModelOutput, Semantics
from src.losses import KullbackLeibler, Loss
from src.losses.build import refuse_an_objective_the_head_does_not_answer
from src.models import Model
from src.tasks import Task
from src.training.learner import StandardLearner
from src.training.registry import learner_registry

DISTILLATION = "distillation"
"""What the term measuring the distance to the teacher is called, in a report and in a declaration.

Named for the method rather than for the divergence it happens to use: a variant measuring that distance
some other way is still this term, and a column that changed name would break every comparison between
the runs on either side of the change.
"""


@learner_registry.register(DISTILLATION)
class DistillationLearner(StandardLearner):
    """A smaller network learns the targets and, beside them, the answers a larger one already gives.

    How far it is from those answers is an objective in its own right, and it is declared in a position
    of its own at ``learner.loss``. What makes two answers comparable at all differs with what the heads
    produce — a projection is softened as it stands, an angle has to be made into a distribution first —
    and none of that is a property of distilling. Whichever objective a run names, the term reports as
    ``distillation``: the column is named for the method, so runs either side of a change of objective
    go on comparing.

    ``losses`` above are the tasks' own, one each, against what the data settled. ``loss`` here is the
    single one this algorithm adds beside them, against what another network answered.

    The teacher is held outside the module tree, in a one-tuple, and that is what keeps it out of
    everything a run writes down. Measured: a module reached that way appears in no ``state_dict``, no
    ``parameters()``, no ``named_children()`` — so the optimizer never sees it, a checkpoint never
    carries it, a summary never prints it, an export never ships it, and the averaging callback never
    averages it. The same measurement is why ``_taught`` moves it by hand: ``.to()`` walks registered
    children, and this is not one.

    How much of the objective each half is stays where such shares already live: the tasks' own
    ``weight`` scales what is learned from the data, and ``weight`` here scales what is learned from the
    teacher. There is no third number mixing them.

    Parameters:
        teacher: The network to learn from, already holding the weights it answers with.
        loss: How the distance to that network's answers is measured. Left out, it is the divergence
            between the two softened, over the projections a head ordinarily produces.
        weight: The share of the objective the teacher's answers are, beside what the targets are worth.
    """

    def __init__(
        self,
        model: Model,
        tasks: Mapping[str, Task],
        losses: Mapping[str, Loss],
        *,
        teacher: Model,
        loss: Loss | None = None,
        weight: float = 1.0,
        learned_only: Collection[str] = (),
    ) -> None:
        super().__init__(model, tasks, losses, learned_only)
        if weight <= 0:
            raise ValueError(
                f"`weight` is the share of the objective the teacher is worth, so it is positive; got {weight}."
            )
        # Outside the module tree on purpose, and the whole reason this is a one-tuple; see the class.
        self._teacher: tuple[Model] = (teacher.eval(),)
        self.loss = loss if loss is not None else KullbackLeibler()
        self._weight = weight
        self._refuse_a_task_this_objective_cannot_read()
        # After the refusal above, which names the objective rather than the column it reports under:
        # the run that needs it wrote no name at all, so the one worth printing is the objective's own.
        # Named for the method rather than for the divergence it uses; see `DISTILLATION`.
        self.loss.log_name = DISTILLATION

    @property
    def teacher(self) -> Model:
        """The network this run learns from, which is not a part of the network it is learning."""
        return self._teacher[0]

    @override
    def _terms(self, output: ModelOutput, batch: Batch) -> list[LossOutput]:
        """The tasks' own objectives, and beside each one how far its answer is from the teacher's.

        A stage that scores no objective scores no agreement either, and the teacher is not run for it:
        a total made of the teacher alone would be a different number under the same name.
        """
        terms = super()._terms(output, batch)
        if not terms:
            return terms
        taught = self._taught(batch)
        return [*terms, *(self._agreement(name, task, output, taught) for name, task in self._scored().items())]

    def _taught(self, batch: Batch) -> ModelOutput:
        """What the teacher answers, on the device the rest of the run has been moved to.

        Moved here rather than once and for all, because a module held outside the tree is not carried
        along by the move that puts a run on its accelerator. Only when it is somewhere else: measured on
        a resnet50, an unconditional move costs 473 us a step against the 27.5 ms its own forward takes.

        Never with gradients. The teacher is a fixed opinion, and a graph built through it every step
        would be thrown away at the end of each.
        """
        device = next(self.parameters()).device
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)
        with torch.no_grad():
            return cast("ModelOutput", self.teacher(batch.inputs))

    def _agreement(self, name: str, task: Task, output: ModelOutput, taught: ModelOutput) -> LossOutput:
        """How far one task's answer is from the teacher's, as a term under that task's own name.

        Summed over the classes and averaged over everything else, so that the number means the same for
        a task answering once per sample and one answering once per pixel. For a flat output that is
        exactly torch's ``batchmean``; for a dense one, ``batchmean`` would report the sum over an image.
        """
        answered, teaches = task.raw(output), task.raw(taught)
        self._refuse_a_teacher_answering_in_another_space(name, answered, teaches)
        return (self.loss(answered, teaches) * self._weight).prefixed(name)

    @staticmethod
    def _refuse_a_teacher_answering_in_another_space(name: str, answered: Tensor, teaches: Tensor) -> None:
        """Two distributions of different widths are not far apart or close; they are not comparable.

        Here rather than at the build, because this is where it can be known: a teacher the root composed
        is sized by the very outputs the student is, but one arriving whole by `_target_` answers with
        whatever it answers with, and only its answer says what that is. `kl_div` broadcasts, so a
        teacher over one class against a student over three gave a finite 53.72823 and a total that
        read like distillation.
        """
        if answered.shape != teaches.shape:
            raise ValueError(
                f"Task {name!r}: this run's network answers {list(answered.shape)} and the teacher "
                f"answers {list(teaches.shape)} — over {teaches.shape[FEATURE_AXIS]} class"
                f"{'' if teaches.shape[FEATURE_AXIS] == 1 else 'es'} against "
                f"{answered.shape[FEATURE_AXIS]}. How far one distribution is from another is a question "
                f"about two of the same width; declare a teacher this run's own tasks size, or one "
                f"trained on the very classes they declare."
            )

    def _refuse_a_task_this_objective_cannot_read(self) -> None:
        """Whether the term can read a task's answer at all, asked of the two things that stop it.

        Softening spreads confidence over classes, and a task answering with something else has none.
        Refused rather than skipped: a run declaring a teacher and quietly learning nothing from it for
        half its tasks reports a total that looks like distillation and is not. What the other semantics
        would need is a different measurement — one score against one score for a number, a divergence
        per label rather than over them for a multilabel answer — and each arrives with its own term.

        Then what the head answers with, against what the objective states it reads. Asked here rather
        than where the run is assembled, because the objective is not always written: left out, it is
        made in this constructor, and a check standing where only the written form is visible would pass
        exactly the configs most likely to be wrong — every one written before the position existed.
        """
        unteachable = sorted(name for name, task in self.tasks.items() if task.semantics is not Semantics.MULTICLASS)
        if unteachable:
            raise ValueError(
                f"{', '.join(unteachable)} cannot be distilled: a teacher is listened to by softening the "
                f"confidence it spread over the classes, and these answer with something that has none. "
                f"Distil the tasks that are judged against one vocabulary, or learn these from their "
                f"targets alone with `learner: {{name: standard}}`."
            )
        for name in self.tasks:
            refuse_an_objective_the_head_does_not_answer(name, self.model.produces(name), self.loss, "learner.loss")
