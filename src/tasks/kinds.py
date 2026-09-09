"""The kinds of task: what each needs — encoder, head, loss, activation, metrics, drawing — in one class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn
from torch.nn.functional import one_hot

from src.core.taxonomy import OutputTopology, Stream
from src.losses import (
    BinaryCrossEntropyCriterion,
    CrossEntropyCriterion,
    ExpectationCriterion,
    InfoNceCriterion,
    MeanSquaredErrorCriterion,
    ProxyAngularCriterion,
    WeightedSumCriterion,
)
from src.models import ConvHead, LinearHead, TaskComponents
from src.tasks.activations import (
    expectation_over,
    identity,
    sigmoid_probabilities,
    softmax_probabilities,
    squeeze_single_output,
)
from src.tasks.adapters import as_class_indices, as_indicators, expectation_of, float_for_loss
from src.tasks.entities import NATIVE_HEAD, Overrides
from src.tasks.registry import task_kind_registry
from src.visualization.annotators import (
    Annotator,
    BinaryReader,
    DenseDrawer,
    GlobalDrawer,
    MulticlassReader,
    MultilabelReader,
    ValueReader,
)
from src.visualization.entities import TaskView

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from src.core.entities import TaskFacts
    from src.core.ports import Backbone, Criterion
    from src.models.composite import Activation, TargetAdapter
    from src.tasks.entities import Task
    from src.visualization.annotators import Drawer, DrawingKnobs, Reader
    from src.visualization.entities import SampleView


class TaskKind(ABC):
    """One familiar kind of task, and everything the framework needs to serve it.

    A kind states its facts as class attributes and builds its parts from the facts
    the data revealed (``TaskFacts``), never from config: the head is sized from the
    class count, a binned regression reads its bins from the encoder that laid them out.
    A kind of your own is a subclass reachable by ``kind: {_target_: my_pkg.Depth}`` (ADR-0003).

    Attributes:
        shape: The structure of one prediction — what readers that branch on shape ask.
        streams: Which backbone streams the head reads, in order; ``None`` defers to the
            backbone's own ``pyramid()``.
        native_by_default: Take the backbone's own head unless a declaration says otherwise —
            the answer while the framework composes no head for the shape.
        default_encoder: Registry name of the encoder a target cell starts from; ``None`` for
            a kind supervised by batch structure rather than by a column.
        default_metrics: What the kind is judged by when the task declares no ``metrics``;
            each entry in the metric grammar, under the label it logs as.
        mixable: Whether a batch transform may blend or stitch this kind's target; how the
            target is made ready for that is ``soften``.
        unavailable: Why a composed model over this kind cannot be trained yet, or ``None``;
            the composition root refuses such a run by name, once the model is built.
            Stage scaffolding for detection; it goes with the criterion (roadmap stages 3–4).
        not_drawn: Why the samples grid shows nothing for this kind, or ``None``.
    """

    shape: ClassVar[OutputTopology] = OutputTopology.GLOBAL
    streams: ClassVar[tuple[str, ...] | None] = (Stream.FEATURES,)
    native_by_default: ClassVar[bool] = False
    default_encoder: ClassVar[str | None] = None
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = {}
    mixable: ClassVar[bool] = True
    unavailable: ClassVar[str | None] = None
    not_drawn: ClassVar[str | None] = None

    @abstractmethod
    def out_features(self, facts: TaskFacts) -> int | None:
        """Head output size; ``None`` where the head projects nothing.

        ``None`` rather than zero: metric learning's embedding *is* the output, so there is no
        width to ask for.

        Raises:
            LookupError: When a fact the size depends on is missing from the task's facts.
        """

    @abstractmethod
    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        """A fresh criterion for one task.

        ``width`` is the first stream the head reads — an embedding size to a criterion that holds
        prototypes (``arcface_proxy``); most kinds ignore it. The same pair a declared ``loss``
        override receives, so a kind's default can be sized exactly as a declaration can.
        """

    @abstractmethod
    def activation(self, facts: TaskFacts) -> Activation:
        """The logits-to-predictions function metrics and inference use."""

    @abstractmethod
    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        """The raw-target shaping for loss and metric views; ``None`` when no column supervises."""

    def metric_kwargs(self, facts: TaskFacts) -> dict[str, Any]:
        """What every metric of this kind is handed (task mode, class counts)."""
        return {}

    def soften(self, target: Tensor, facts: TaskFacts) -> Tensor:
        """One target in the shape a weighted sum of two targets needs.

        A number, an indicator vector or a distribution already admits a weighted sum and
        only wants a float dtype; a kind whose target is a class index overrides this to
        widen it first.
        """
        return target.float()

    def head(self, in_features: int | tuple[int, ...], out_features: int | None) -> nn.Module:
        """A fresh head at the resolved sizes: one linear projection unless a kind says otherwise."""
        width = _one_width(in_features, self)
        if out_features is None:
            raise ValueError(
                f"{type(self).__name__} projects onto classes, so its head needs a width; none was asked for."
            )
        return LinearHead(width, out_features)

    def components(self, task: Task, backbone: Backbone, overrides: Overrides | None = None) -> TaskComponents:
        """The parts that serve ``task`` inside a composite model, sized from the backbone and the facts.

        Declared wins over derived, as everywhere: a stream, a head or a loss the task
        declared replaces the kind's answer. How many streams a head can read is the head's
        own business — a single-stream head handed several refuses itself by name.

        Raises:
            LookupError: For a native head the backbone does not offer, a pyramid it does not
                declare, or a fact the sizes need that the task's facts lack.
        """
        declared = overrides if overrides is not None else Overrides()
        streams = declared.streams or self.streams or backbone.pyramid()
        if not streams:
            raise LookupError(
                f"{type(backbone).__name__} declares no pyramid, and task '{task.name}' reads one: a "
                f"'{self.shape}' task needs a backbone with a detection head of its own."
            )
        widths = tuple(backbone.feature_dim(name) for name in streams)
        in_features: int | tuple[int, ...] = widths[0] if len(widths) == 1 else widths
        try:
            out_features = self.out_features(task.facts)
        except LookupError as error:
            raise LookupError(f"Task '{task.name}': {error}") from None
        head: nn.Module
        chosen = declared.head
        if chosen == NATIVE_HEAD or (chosen is None and self.native_by_default):
            native = backbone.native_head(streams, in_features, _projected(task, out_features))
            if native is None:
                raise LookupError(f"{type(backbone).__name__} offers no native head for {', '.join(streams)}.")
            # Used as it is: a wrapper would bury contract paths (freeze's `heads.<task>.base`, a
            # checkpoint's `heads.<task>.weight`) under a private attribute.
            head = native
        elif chosen is None:
            head = self.head(in_features, out_features)
        else:
            head = chosen(in_features, _projected(task, out_features))
        criterion = (
            declared.loss(task.facts, widths[0]) if declared.loss is not None else self.loss(task.facts, widths[0])
        )
        return TaskComponents(
            head=head,
            criterion=criterion,
            activation=self.activation(task.facts),
            target_adapter=self.target_adapter(task.facts),
            streams=streams,
            weight=task.weight,
        )

    def reader(self, knobs: DrawingKnobs) -> Reader:
        """How this kind's activated outputs and targets are read for a page."""
        raise TypeError(self.not_drawn or f"{type(self).__name__} declares no reader for the samples grid.")

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        """How a pair of readings becomes labels and a verdict on a page."""
        raise TypeError(self.not_drawn or f"{type(self).__name__} declares no drawer for the samples grid.")

    def annotate(
        self, view: SampleView, task: Task, outputs: Tensor, targets: Tensor, index: int, knobs: DrawingKnobs
    ) -> None:
        """Label batch element ``index`` on ``view``; ``outputs`` are the task's activated outputs."""
        if self.not_drawn is not None:
            raise TypeError(f"Task '{task.name}' ({type(self).__name__}): {self.not_drawn}")
        subject = TaskView(task.name, task.facts.class_names)
        Annotator(self.reader(knobs), self.drawer(knobs)).annotate(view, subject, outputs, targets, index)


def _one_width(in_features: int | tuple[int, ...], kind: TaskKind) -> int:
    """The width of the one stream a single-stream kind's head is sized from."""
    if isinstance(in_features, int):
        return in_features
    raise ValueError(
        f"{type(kind).__name__} reads one stream, but was sized from {len(in_features)} widths "
        f"{in_features}; a head over several streams belongs to a kind that reads them."
    )


def _projected(task: Task, out_features: int | None) -> int:
    """The width a *declared* head is built at, refused where the task projects nothing.

    ``out_features is None`` is metric learning's contract — the embedding is already the
    output — and only the kind's own head knows to answer that with an identity. A head
    named in config does not: as a zero it reached ``CosineHead(in_features, 0)`` and
    built a classifier with no prototypes, failing several frames from the declaration.
    """
    if out_features is None:
        raise ValueError(
            f"Task '{task.name}' is supervised by comparison, so its embedding is the output and a "
            f"declared head has nothing to project onto. Drop 'head' from the task."
        )
    return out_features


# --- the pieces kinds are made of --------------------------------------------------------------

CLASSIFICATION_METRICS: Mapping[str, Mapping[str, Any]] = {
    # `average="none"` asks for the per-class vector; torchmetrics' binary task ignores it
    # and answers with the positive class's scalar (measured), so one table serves all three.
    "f1": {"name": "f1", "average": "none"},
    "precision": {"name": "precision", "average": "none"},
    "recall": {"name": "recall", "average": "none"},
    "confusion_matrix": {"name": "confusion_matrix", "normalize": "true"},
}
# A multilabel confusion matrix is one small matrix per label, which the metric publishes
# as nothing (see ``ConfusionMatrixMetric``); a default that never reports is not declared.
MULTILABEL_METRICS: Mapping[str, Mapping[str, Any]] = {
    label: spec for label, spec in CLASSIFICATION_METRICS.items() if label != "confusion_matrix"
}
# Per-pixel per-class f1 IS dice (2TP/(2TP+FP+FN)), so the customary dice score is
# already on this list under f1's name; iou adds the strict-overlap reading.
SEGMENTATION_METRICS: Mapping[str, Mapping[str, Any]] = {
    "iou": {"name": "iou", "average": "none"},
    **CLASSIFICATION_METRICS,
}
MULTILABEL_SEGMENTATION_METRICS: Mapping[str, Mapping[str, Any]] = {
    "iou": {"name": "iou", "average": "none"},
    **MULTILABEL_METRICS,
}


class MulticlassLabels:
    """Exactly one class per prediction: cross-entropy over softmax probabilities."""

    default_encoder: ClassVar[str | None] = "label"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = CLASSIFICATION_METRICS

    def out_features(self, facts: TaskFacts) -> int | None:
        if facts.num_classes is None:
            raise LookupError("needs num_classes; run setup before building tasks.")
        return facts.num_classes

    def soften(self, target: Tensor, facts: TaskFacts) -> Tensor:
        """A class index widens to one-hot at the class count the facts carry."""
        if facts.num_classes is None:
            raise LookupError("needs num_classes to widen a class index; run setup first.")
        return one_hot(target.long(), facts.num_classes).float()

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return CrossEntropyCriterion()

    def activation(self, facts: TaskFacts) -> Activation:
        return softmax_probabilities

    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        return as_class_indices

    def metric_kwargs(self, facts: TaskFacts) -> dict[str, Any]:
        return {"task": "multiclass", "num_classes": facts.num_classes}

    def reader(self, knobs: DrawingKnobs) -> Reader:
        return MulticlassReader()


class BinaryLabels:
    """A single yes-or-no probability: BCE on one logit, sigmoid for metrics."""

    default_encoder: ClassVar[str | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = CLASSIFICATION_METRICS

    def out_features(self, facts: TaskFacts) -> int | None:
        return 1

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return BinaryCrossEntropyCriterion()

    def activation(self, facts: TaskFacts) -> Activation:
        return sigmoid_probabilities

    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        return as_indicators

    def metric_kwargs(self, facts: TaskFacts) -> dict[str, Any]:
        return {"task": "binary"}

    def reader(self, knobs: DrawingKnobs) -> Reader:
        return BinaryReader(threshold=knobs.threshold)


class MultilabelLabels:
    """Independent per-class probabilities: BCE per class, sigmoid for metrics."""

    default_encoder: ClassVar[str | None] = "multilabel"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = MULTILABEL_METRICS

    def out_features(self, facts: TaskFacts) -> int | None:
        if facts.num_classes is None:
            raise LookupError("needs num_classes; run setup before building tasks.")
        return facts.num_classes

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return BinaryCrossEntropyCriterion()

    def activation(self, facts: TaskFacts) -> Activation:
        return sigmoid_probabilities

    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        return as_indicators

    def metric_kwargs(self, facts: TaskFacts) -> dict[str, Any]:
        return {"task": "multilabel", "num_labels": facts.num_classes}

    def reader(self, knobs: DrawingKnobs) -> Reader:
        return MultilabelReader(threshold=knobs.threshold)


class DenseOutput:
    """One prediction per spatial location, projected from the decoder stream.

    A dense cell is a mask file whatever the labels mean, so the encoder and the metrics
    are this shape's, ahead of the semantics a kind pairs it with.
    """

    shape: ClassVar[OutputTopology] = OutputTopology.DENSE
    streams: ClassVar[tuple[str, ...] | None] = (Stream.DECODER,)
    default_encoder: ClassVar[str | None] = "mask"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = SEGMENTATION_METRICS

    def head(self, in_features: int | tuple[int, ...], out_features: int | None) -> nn.Module:
        width = _one_width(in_features, self)  # type: ignore[arg-type]  # the mixin is always on a kind
        if out_features is None:
            raise ValueError("A dense head projects onto classes, so it needs a width; none was asked for.")
        return ConvHead(width, out_features)

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        return DenseDrawer(ignore_index=knobs.ignore_index)


# --- the kinds -----------------------------------------------------------------------------------


@task_kind_registry.register("classification")
class Classification(MulticlassLabels, TaskKind):
    """One class per sample, read off the pooled features."""

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        return GlobalDrawer()


@task_kind_registry.register("binary_classification")
class BinaryClassification(BinaryLabels, TaskKind):
    """Yes or no per sample."""

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        return GlobalDrawer()


@task_kind_registry.register("multilabel_classification")
class MultilabelClassification(MultilabelLabels, TaskKind):
    """Any number of classes per sample."""

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        return GlobalDrawer()


@task_kind_registry.register("regression")
class Regression(TaskKind):
    """A number per sample, learned directly or through bins.

    Which of the two is in play is not a second declaration: it follows from the target
    encoder. A plain encoder yields one output against mean squared error; a binned one
    reports ``class_values``, and the same number is then learned as a distribution —
    cross-entropy over the bins plus a term on the expectation — and read back as the
    value it stands for. Either way the metrics compare numbers.
    """

    default_encoder: ClassVar[str | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = {"mae": {"name": "mae"}}

    def out_features(self, facts: TaskFacts) -> int | None:
        return 1 if facts.class_values is None else len(facts.class_values)

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        if facts.class_values is None:
            return MeanSquaredErrorCriterion()
        # Cross-entropy leads: it shapes the whole distribution. The expectation term is a
        # correction that keeps the reported number aligned with the metric, and is
        # deliberately the lighter of the two — on its own, any distribution with the
        # right mean satisfies it. Config overrides both weights at once.
        return WeightedSumCriterion([(CrossEntropyCriterion(), 1.0), (ExpectationCriterion(facts.class_values), 0.5)])

    def activation(self, facts: TaskFacts) -> Activation:
        return squeeze_single_output if facts.class_values is None else expectation_over(facts.class_values)

    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        return float_for_loss if facts.class_values is None else expectation_of(facts.class_values)

    def reader(self, knobs: DrawingKnobs) -> Reader:
        return ValueReader()

    def drawer(self, knobs: DrawingKnobs) -> Drawer:
        return GlobalDrawer()


@task_kind_registry.register("metric_learning")
class MetricLearning(TaskKind):
    """Embeddings shaped against labels: the backbone's pooled features, and learnable class proxies.

    The output already holds final embeddings, so the head is identity and there is no width
    to ask for; the criterion holds the proxies (``arcface_proxy``), sized by the vocabulary and the
    embedding width, so the task declares ``classes`` and a label column like a classification does.
    Comparison against the batch instead of against labels is ``contrastive`` or ``ranking``. Soft
    labels break its losses, so no batch transform may touch it, and there is no per-sample label
    to draw.
    """

    mixable: ClassVar[bool] = False
    not_drawn: ClassVar[str | None] = "metric learning has no per-sample label to show"
    default_encoder: ClassVar[str | None] = "label"

    def out_features(self, facts: TaskFacts) -> int | None:
        return None

    def head(self, in_features: int | tuple[int, ...], out_features: int | None) -> nn.Module:
        _one_width(in_features, self)  # one stream, or refused by name
        return nn.Identity()  # the embedding is the output

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return ProxyAngularCriterion.sized(facts, width)

    def activation(self, facts: TaskFacts) -> Activation:
        return identity

    def target_adapter(self, facts: TaskFacts) -> TargetAdapter | None:
        return as_class_indices if facts.num_classes is not None else None


@task_kind_registry.register("contrastive")
class Contrastive(MetricLearning):
    """Two encoders, one aligned embedding per stream (CLIP-style): InfoNCE over the stacked views, no target.

    Supervision is the batch's own structure — the in-batch diagonal — so there is no column to
    encode and no vocabulary; the head reads the ``embeddings`` stream the model stacks.
    """

    streams: ClassVar[tuple[str, ...] | None] = (Stream.EMBEDDINGS,)
    default_encoder: ClassVar[str | None] = None

    def loss(self, facts: TaskFacts, width: int) -> Criterion:
        return InfoNceCriterion()


@task_kind_registry.register("ranking")
class Ranking(Contrastive):
    """N views of one item through a shared encoder — served exactly as ``contrastive``; the model differs, not the task."""


@task_kind_registry.register("segmentation")
class Segmentation(DenseOutput, MulticlassLabels, TaskKind):
    """One class per pixel."""


@task_kind_registry.register("binary_segmentation")
class BinarySegmentation(DenseOutput, BinaryLabels, TaskKind):
    """Foreground or not, per pixel."""


@task_kind_registry.register("multilabel_segmentation")
class MultilabelSegmentation(DenseOutput, MultilabelLabels, TaskKind):
    """Any number of classes per pixel — declared, and refused at build until an encoder serves it.

    The kind's BCE reads one plane per class, a multi-hot ``[C, H, W]``; the ``mask`` encoder it
    inherits yields an index map, one class per pixel, and no shipped encoder produces the other.
    ``unavailable`` says so when the model is built, by name, instead of at the first step.
    """

    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = MULTILABEL_SEGMENTATION_METRICS
    unavailable: ClassVar[str | None] = (
        "no shipped encoder produces its target, a multi-hot [C, H, W] per pixel — 'mask' yields an index "
        "map, one class per pixel, and the kind's BCE reads one plane per class. Declare 'segmentation' "
        "for one class per pixel, or bring an encoder that reads several."
    )


@task_kind_registry.register("detection")
class Detection(MulticlassLabels, TaskKind):
    """A variable-length set of boxes per sample, each with one class of N.

    The framework composes no detection head: the backbone's native head serves, reading
    the pyramid the backbone declares and sized by the facts like every head. The loss
    inherited here is the stage-2 placeholder that lets the model *build*; ``unavailable``
    is what keeps a run from training it before the criterion lands (roadmap stages 3–4), and both go
    together.
    """

    shape: ClassVar[OutputTopology] = OutputTopology.INSTANCES
    streams: ClassVar[tuple[str, ...] | None] = None  # the backbone's pyramid: its levels, its count, its order
    native_by_default: ClassVar[bool] = True
    default_encoder: ClassVar[str | None] = "boxes"
    default_metrics: ClassVar[Mapping[str, Mapping[str, Any]]] = {"map": {"name": "map"}}
    mixable: ClassVar[bool] = False
    unavailable: ClassVar[str | None] = (
        "a composed detection model builds, but has no criterion yet (detection roadmap, stages 3–4), so nothing "
        "can train it."
    )
    not_drawn: ClassVar[str | None] = "a set of objects is not drawn yet"
