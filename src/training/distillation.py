"""Learning from a second network as well as from the data: what it answers is the other half of the objective."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast, override

import torch
from torch.nn.functional import kl_div, log_softmax

from src.core import FEATURE_AXIS, Batch, LossOutput, ModelOutput, Semantics
from src.losses import Loss
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

    What the second network buys is everything the targets leave out. A label says one class is right;
    a trained teacher says how wrong each of the others is, and those proportions are what a small
    network cannot work out from few examples. So the term here is a divergence between two whole
    distributions rather than a second look at the true class.

    Both are softened before they are compared, because the distinctions worth learning sit in the small
    probabilities, and an unsoftened teacher spends all of its confidence on one class. Softening also
    shrinks the term, which is why it is scaled back by the square of the temperature: measured on random
    logits, without that scale the gradient falls fourfold per doubling of the temperature, so ``weight``
    would silently mean less at every step of it.

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
        temperature: How far both distributions are softened before they are compared. One compares them
            as they are; the usual range is two to ten.
        weight: The share of the objective the teacher's answers are, beside what the targets are worth.
    """

    def __init__(
        self,
        model: Model,
        tasks: Mapping[str, Task],
        losses: Mapping[str, Loss],
        *,
        teacher: Model,
        temperature: float = 4.0,
        weight: float = 1.0,
        learned_only: Collection[str] = (),
    ) -> None:
        super().__init__(model, tasks, losses, learned_only)
        if temperature <= 0:
            raise ValueError(
                f"A temperature softens a distribution by dividing by it, so it is positive; got {temperature}."
            )
        if weight <= 0:
            raise ValueError(
                f"`weight` is the share of the objective the teacher is worth, so it is positive; got {weight}."
            )
        self._refuse_a_task_with_no_distribution_to_soften()
        # Outside the module tree on purpose, and the whole reason this is a one-tuple; see the class.
        self._teacher: tuple[Model] = (teacher.eval(),)
        self._temperature = temperature
        self._weight = weight

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
        softened = log_softmax(task.raw(output) / self._temperature, dim=FEATURE_AXIS)
        teaching = log_softmax(task.raw(taught) / self._temperature, dim=FEATURE_AXIS)
        pointwise = kl_div(softened, teaching, reduction="none", log_target=True)
        divergence = pointwise.sum(FEATURE_AXIS).mean() * self._temperature**2
        return (LossOutput.reported(DISTILLATION, divergence) * self._weight).prefixed(name)

    def _refuse_a_task_with_no_distribution_to_soften(self) -> None:
        """Softening spreads confidence over classes, and a task answering with something else has none.

        Refused rather than skipped: a run declaring a teacher and quietly learning nothing from it for
        half its tasks reports a total that looks like distillation and is not. What the other semantics
        would need is a different measurement — one score against one score for a number, a divergence
        per label rather than over them for a multilabel answer — and each arrives with its own term.
        """
        unteachable = sorted(name for name, task in self.tasks.items() if task.semantics is not Semantics.MULTICLASS)
        if unteachable:
            raise ValueError(
                f"{', '.join(unteachable)} cannot be distilled: a teacher is listened to by softening the "
                f"confidence it spread over the classes, and these answer with something that has none. "
                f"Distil the tasks that are judged against one vocabulary, or learn these from their "
                f"targets alone with `learner: {{name: standard}}`."
            )
