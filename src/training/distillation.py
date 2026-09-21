"""Learning from a second network as well as from the data: what it answers is the other half of the objective."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import cast, override

import torch
from torch import Tensor

from src.core import FEATURE_AXIS, Batch, LossOutput, ModelOutput, Semantics, TensorTree, as_children, require_tensor
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


REPRESENTATION = "representation"
"""What the term measuring the distance between the two networks' features is called.

Its own word rather than ``DISTILLATION``, because the two measure different things: one is a
divergence between answers, the other a distance between the features those answers were read from.
Under one word a chart would carry two lines a reader could not tell apart, and the prefix — a task
for one, a stream for the other — is not something a reader of a legend can be expected to decode.
"""


@learner_registry.register(DISTILLATION)
class DistillationLearner(StandardLearner):
    """A smaller network learns the targets and, beside them, what a larger one already answers and reads.

    What it learns from that second network is declared in a position of its own at ``learner.loss`` —
    one term or several, written the way a task's losses are. A term names a ``stream`` to be compared
    with the teacher's feature of that name, and names none to be compared with its answers. The two are
    indexed differently, which is why they are held apart here: one term per task for the answers, one
    per stream for the features.

    What makes two answers comparable at all differs with what the heads produce — a projection is
    softened as it stands, an angle has to be made into a distribution first — and none of that is a
    property of distilling. A term reports under the name its *reading* gives it, ``distillation`` or
    ``representation``, so that runs either side of a change of measure go on comparing; a run writing
    ``log_name`` keeps what it wrote, and nothing here renames an objective it was handed.

    Both halves are worth something on their own. Answers carry the proportions a label leaves out —
    how wrong each of the other classes is — and a teacher confident enough spends nearly all of that
    on one class, leaving little to learn. Features carry the representation those answers were read
    from, which is what a frozen head turns into a shared space: a student whose features land where the
    teacher's do answers as the teacher does, by construction. Neither is the other, and a run may write
    one, the other, or both.

    ``losses`` above are the tasks' own, one each, against what the data settled. What is written here
    is what this algorithm adds beside them, against what another network answered or read.

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
        loss: How the distance to that network's *answers* is measured, one term per task. Left out with
            nothing else written, it is the divergence between the two softened, over the projections a
            head ordinarily produces; left out beside a term over features, it is nothing, because a run
            that wrote its terms gets those and no other.
        representation: How the distance to that network's *features* is measured, by stream. Two
            networks publishing a stream of one name and one width are comparable in it, whether they
            arrived at that width by construction or through a ``model.neck``.
        weight: The share of the objective everything learned from the teacher is worth, beside what the
            targets are worth. It scales both halves; the share of one term within them is its own.
    """

    def __init__(
        self,
        model: Model,
        tasks: Mapping[str, Task],
        losses: Mapping[str, Loss],
        *,
        teacher: Model,
        loss: Loss | None = None,
        representation: Mapping[str, Loss] | None = None,
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
        # Registered as children so the terms travel with the run: `.to()` walks them, a checkpoint keeps
        # them, and an objective holding parameters is kept where every other one is. Measured: an empty
        # `ModuleDict` puts nothing in a `state_dict`, so a run comparing no features writes down no more
        # than it ever did.
        self.representation = as_children(dict(representation or {}))
        self.loss = loss
        if self.loss is None and not self.representation:
            # Nothing written at all: an algorithm that learns from a second network learns from its
            # answers, which is what distilling has meant here. Not added beside terms a run did write —
            # one pulling features alone would then descend a divergence nobody asked for. Named here
            # because no declaration named it; a term that was written keeps the name it was given, which
            # is how a run tells two terms over one reading apart.
            self.loss = KullbackLeibler()
            self.loss.log_name = DISTILLATION
        self._weight = weight
        self._refuse_a_task_this_objective_cannot_read()

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
        objective = self.loss
        answered = (
            [self._agreement(name, task, objective, output, taught) for name, task in self._scored().items()]
            if objective is not None
            else []
        )
        aligned = [self._alignment(stream, output, taught) for stream in self.representation]
        return [*terms, *answered, *aligned]

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

    def _agreement(
        self, name: str, task: Task, objective: Loss, output: ModelOutput, taught: ModelOutput
    ) -> LossOutput:
        """How far one task's answer is from the teacher's, as a term under that task's own name.

        Summed over the classes and averaged over everything else, so that the number means the same for
        a task answering once per sample and one answering once per pixel. For a flat output that is
        exactly torch's ``batchmean``; for a dense one, ``batchmean`` would report the sum over an image.
        """
        answered, teaches = task.raw(output), task.raw(taught)
        self._refuse_a_teacher_answering_in_another_space(name, answered, teaches)
        return (objective(answered, teaches) * self._weight).prefixed(name)

    def _alignment(self, stream: str, output: ModelOutput, taught: ModelOutput) -> LossOutput:
        """How far this run's features are from the teacher's, as a term under the stream's own name.

        Prefixed by the stream as an agreement is prefixed by the task, because that is what tells one
        term of this kind from another: a run may pull two streams, and a total that summed them would
        answer neither `how far is the encoder` nor `how far is the decoder`.
        """
        answered = require_tensor(_published(output, stream, "this run's network"), name=stream)
        teaches = require_tensor(_published(taught, stream, "the teacher"), name=stream)
        self._refuse_a_teacher_whose_features_are_another_shape(stream, answered, teaches)
        objective = cast("Loss", self.representation[stream])
        return (objective(answered, teaches) * self._weight).prefixed(stream)

    @staticmethod
    def _refuse_a_teacher_whose_features_are_another_shape(stream: str, answered: Tensor, teaches: Tensor) -> None:
        """Two representations of different widths are not close or far apart; they are not comparable.

        Here rather than at the build, for the reason the refusal below keeps: a teacher the root composed
        publishes what its own backbone declares, but one arriving whole by `_target_` publishes whatever
        it publishes, and only its answer says what that is. `mse_loss` broadcasts, so a teacher of one
        number against a student of a hundred gives a finite number and a total that reads like work.
        """
        if answered.shape != teaches.shape:
            raise ValueError(
                f"Stream {stream!r}: this run's network publishes {list(answered.shape)} and the teacher "
                f"publishes {list(teaches.shape)}. How far one representation is from another is a "
                f"question about two of the same shape; bring both to one width with a `model.neck`, "
                f"or compare a stream they already publish alike."
            )

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
        if self.loss is None:
            return
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


def _published(answered: ModelOutput, stream: str, whose: str) -> TensorTree:
    """The stream a term names, refused by name where the network that was asked publishes others.

    Named here rather than left to a `KeyError` for the reason every such refusal is: three declarations
    have to agree on one word — the two backbones' published names and the `stream` a term wrote — and a
    bare key error names none of them.
    """
    try:
        return answered.features[stream]
    except KeyError:
        carried = ", ".join(sorted(answered.features)) or "nothing"
        raise ValueError(
            f"A term of `learner.loss` compares stream {stream!r}, and {whose} publishes {carried}. A "
            f"feature is compared between two networks that both publish it: write the name they share, "
            f"or bring one of them to it with a `model.neck`."
        ) from None
