"""What a run learns for one target: the class is the semantics, the instance is this run's facts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import ClassVar

from torch import Tensor

from src.core import (
    SPATIAL,
    Axis,
    Batch,
    ModelOutput,
    Representation,
    Semantics,
    Stream,
    TargetInfo,
    TensorShape,
    TensorTree,
    require_tensor,
)

type LossDeclaration = str | Mapping[str, object] | Sequence[Mapping[str, object]]
type TargetEncoderDeclaration = str | Mapping[str, object]
"""How a kind names the encoder its column starts from: a registry name, or a whole declaration.

Both, for the same reason ``default_head`` is a whole declaration: a kind of one's own reaches an
encoder of one's own by import path, without registering it into this tree or repeating the line in
every experiment that names the kind."""
"""A loss as a task declares its default: a registry name, one declaration, or several to weigh together."""


class Task(ABC):
    """One learning objective: its semantics as a subclass, its name, facts and weight as an instance.

    A task holds no modules. It declares what a run should assemble around it — which encoder reads its
    column, which head serves it, what it is judged by — and it converts between the three views one step
    needs: what the loss compares, what the model's output means, and what a metric scores.

    ``weight`` is its share of the objective and ``lr`` the rate its own parameters move at: both are
    facts of this run rather than of the semantics, which is why they sit on the instance. They arrive
    already checked — `tasks.<name>` is a typed section — so nothing here reads them twice.

    Attributes:
        default_head: The head a task gets when a run declares none, and the stream it reads.
        default_target_encoder: The encoder its column starts from — a registry name, or a whole
            declaration for one this framework does not hold — or None when the batch itself is the
            supervision. Read before the data is prepared, so it cannot see facts.
        default_metrics: What the task is judged by when a run declares no metrics of its own.
        semantics: What this task's labels mean, where they mean one of the three things a vocabulary
            can mean; None where the target is a number rather than a label.
    """

    output_axis: ClassVar[str] = Axis.CLASSES
    """Which axis of this task's output carries its width — the one a head is built at.

    Classes for anything read against a vocabulary, which is every kind but one; a kind whose answer is
    a direction says so here, and the shape it produces follows without restating the width.
    """

    publishes: ClassVar[Representation] = Representation.PROJECTED
    """What this kind answers with once it has read a projection as what that projection means.

    The word a deployment is handed for the numbers it reads back, which no tensor carries. It is paired
    with ``publish`` below and the two are declared together: the default reads a projection as nothing,
    so what such a kind answers with *is* the projection, and saying so is true rather than a fallback.
    """

    default_head: ClassVar[Mapping[str, object]] = {"name": "linear", "stream": Stream.POOLED}
    default_target_encoder: ClassVar[TargetEncoderDeclaration | None] = None
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {}
    semantics: ClassVar[Semantics | None] = None

    def __init__(self, name: str, info: TargetInfo, *, weight: float = 1.0, lr: float | None = None) -> None:
        self.name = name
        self.info = info
        self.weight = weight
        self.lr = lr

    @abstractmethod
    def out_features(self) -> int:
        """How many values the model produces per position — per sample, or per pixel for a dense task.

        An instance method because the width is not always a fact of the target: an embedding's is a
        choice of the model, declared on the kind and read from ``self``.
        """

    def output_shape(self) -> TensorShape:
        """The shape one prediction has, which is what a head is built to produce."""
        return TensorShape(axes=(self.output_axis,), sizes=(self.out_features(),))

    @property
    def dense(self) -> bool:
        """Whether this task decides at every pixel rather than once for the whole sample.

        Read off the shape it produces rather than declared a second time: a mixing transform refuses
        such a task and a page draws it as masks instead of chips, and both ask the same question.
        """
        return bool(SPATIAL & set(self.output_shape().axes))

    @property
    def embeds(self) -> bool:
        """Whether this task answers with a direction in a space rather than a reading of the sample.

        Read off the shape it produces, as ``dense`` is, and for the same reason: a page has no cell to
        draw one in and a deployment record has no vocabulary to publish for one, and both ask this
        same question.
        """
        return Axis.EMBEDDING in self.output_shape().axes

    @property
    @abstractmethod
    def default_loss(self) -> LossDeclaration:
        """The objective this task is learned by when a run declares none; instance-level, so it can read facts."""

    def facts(self) -> Mapping[str, object]:
        """What this run settled about the target, for whoever is built around it to name in its constructor.

        One answer for losses and for metrics alike: a fact reaches a constructor that names it, and a
        declaration restating one is refused. A new fact is a key here, not an edit to either builder.
        """
        return {"semantics": self.semantics, "num_classes": self.info.num_classes, "values": self.info.values}

    @abstractmethod
    def loss_target(self, batch: Batch) -> Tensor:
        """The target as the loss compares it; a batch transform may have left it soft."""

    @abstractmethod
    def metric_view(self, batch: Batch) -> Tensor:
        """The same target as a metric scores it — hard where the loss reads a share."""

    def postprocess(self, output: ModelOutput, produced: Representation) -> TensorTree:
        """What this task's numbers mean, given what the network made of them. Never the loss's view.

        A projection is read as what this kind means; numbers that are *already* a reading are handed
        on as they stand, because nothing this side of the run can turn them into another one. Measured
        on the arrangement `examples/metric_learning.yaml` documents — 37 breeds, a ``cosine`` head
        under ``arcface`` — a softmax over cosines answers 0.0720 for a converged sample and can never
        exceed 0.1703 (0.0073 at a thousand identities), while the objective reads those very numbers
        at 1.0000. The difference is a temperature the objective holds and a task cannot see; borrowing
        it would put a training hyperparameter into an artifact as a calibration nobody measured.

        Concrete rather than abstract, and the one home for the rule ``answers_with`` states in words:
        a kind that overrides this instead of ``publish`` takes the numbers and the word that describes
        them apart, and the record shipped beside an artifact is the half that then lies.
        """
        raw = self.raw(output)
        return self.publish(raw) if produced == Representation.PROJECTED else raw

    def answers_with(self, produced: Representation) -> Representation:
        """The word for what ``postprocess`` just answered with, for the record beside an artifact.

        The other half of the rule above, in the only form a record can carry: a tensor says nothing
        about which of these it holds, which is the whole reason ``Representation`` exists. A reading
        the network already made keeps the word it arrived under, exactly as the numbers do.

        Compared by value rather than by identity, as every reading in this framework is: a network a
        run wrote itself may spell ``produces = "projected"`` and be taken at its word.
        """
        return self.publishes if produced == Representation.PROJECTED else produced

    def publish(self, projected: Tensor) -> TensorTree:
        """This kind's reading of a projection: a share per class, a number, a direction, a mask.

        Handed the projection alone rather than the output it came in, so a kind cannot reach past its
        own task. The default reads it as nothing, which is what ``publishes`` above says it answers
        with — a pair, and a kind changing one of them without the other is what the contract test over
        every kind exists to catch.
        """
        return projected

    def soften(self, target: Tensor) -> Tensor:
        """This task's target in the shape a weighted sum of two of them needs.

        A number, an indicator vector or a distribution already admits one and wants only a float
        dtype. A target that is a class *index* does not — an average of indices names a third class
        that neither sample was — so the kinds whose targets are indices widen them first.
        """
        return target.float()

    def target(self, batch: Batch) -> Tensor:
        """This task's raw target, refused by name when the batch carries none."""
        return self._own(batch.targets, "target in the batch")

    def raw(self, output: ModelOutput) -> Tensor:
        """What this task's head produced, before it means anything."""
        return self._own(output.outputs, "output from the model")

    def _own(self, values: Mapping[str, TensorTree], what: str) -> Tensor:
        try:
            return require_tensor(values[self.name], name=self.name)
        except KeyError:
            held = ", ".join(sorted(values)) or "none"
            raise LookupError(f"No {what} for task {self.name!r}; there is: {held}.") from None
