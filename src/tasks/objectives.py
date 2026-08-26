"""How each label semantics is learned and evaluated: the components per ``Objective`` member."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, override

from src.core.taxonomy import Objective
from src.losses import (
    BinaryCrossEntropyCriterion,
    CrossEntropyCriterion,
    ExpectationCriterion,
    InfoNceCriterion,
    MeanSquaredErrorCriterion,
    WeightedSumCriterion,
)
from src.tasks.activations import (
    expectation_over,
    identity,
    sigmoid_probabilities,
    softmax_probabilities,
    squeeze_single_output,
)
from src.tasks.adapters import as_class_indices, as_indicators, expectation_of, float_for_loss
from src.tasks.registry import objective_registry

if TYPE_CHECKING:
    from src.core.entities import TargetFacts
    from src.core.ports import Activation, Criterion, TargetAdapter


class TaskObjective(ABC):
    """The behaviour behind one ``Objective`` member: head size, criterion, activation, adapter.

    Every component is built from ``TargetFacts``, so how a target is *represented* stays
    the encoder's choice rather than a second axis: the same continuous semantics is served
    by one output against MSE, or by bins against cross-entropy, depending on the facts.
    ``needs_num_classes`` tells the builder a class count must be there.
    """

    needs_num_classes: ClassVar[bool] = False

    @abstractmethod
    def out_features(self, facts: TargetFacts) -> int | None:
        """Head output size for this label semantics; ``None`` where a head projects nothing.

        ``None`` rather than zero: metric learning's embedding *is* the output, so there is no
        width to ask for.
        """

    @abstractmethod
    def build_criterion(self, facts: TargetFacts) -> Criterion:
        """A fresh criterion instance for one task."""

    @abstractmethod
    def build_activation(self, facts: TargetFacts) -> Activation:
        """The logits-to-predictions function metrics and inference use."""

    @abstractmethod
    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter | None:
        """The raw-target shaping for loss and metric views.

        ``None`` for structure-supervised objectives: no target column means
        there is nothing to adapt.
        """

    def metric_kwargs(self, facts: TargetFacts) -> dict[str, Any]:
        """Base kwargs every metric of this objective receives (task mode, class counts)."""
        return {}

    default_target_encoder: ClassVar[str | None] = None
    """Registry name of the encoder shaping a scalar-ish cell of this semantics.

    A name rather than an instance: an objective knows which *form* its loss needs — a
    class index, a float, an indicator vector — while the ``target_encoder_registry``
    stays the one place that knows how to build it, so this package never reaches into
    the data layer. ``None`` for objectives supervised by batch structure rather than by
    a column.

    A constant, not a method: the choice is made before any data is read, because the
    facts an encoder could depend on are exactly what that encoder produces at ``fit``.
    Its neighbour ``needs_num_classes`` already shows the honest form. The *shape* of a
    cell outranks its semantics, so ``TaskTopology`` carries the same declaration and
    ``default_target_encoder`` in the builder composes the two.
    """


@objective_registry.register_instance(Objective.MULTICLASS)
class MulticlassObjective(TaskObjective):
    """Exactly one class per prediction: cross-entropy over softmax probabilities."""

    needs_num_classes: ClassVar[bool] = True
    default_target_encoder: ClassVar[str | None] = "label"

    def out_features(self, facts: TargetFacts) -> int:
        if facts.num_classes is None:
            raise LookupError("Multiclass needs num_classes; profile the data before building tasks.")
        return facts.num_classes

    def build_criterion(self, facts: TargetFacts) -> Criterion:
        return CrossEntropyCriterion()

    def build_activation(self, facts: TargetFacts) -> Activation:
        return softmax_probabilities

    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter:
        return as_class_indices

    @override
    def metric_kwargs(self, facts: TargetFacts) -> dict[str, Any]:
        return {"task": "multiclass", "num_classes": facts.num_classes}


@objective_registry.register_instance(Objective.BINARY)
class BinaryObjective(TaskObjective):
    """A single yes-or-no probability: BCE on one logit, sigmoid for metrics."""

    default_target_encoder: ClassVar[str | None] = "scalar"

    def out_features(self, facts: TargetFacts) -> int:
        return 1

    def build_criterion(self, facts: TargetFacts) -> Criterion:
        return BinaryCrossEntropyCriterion()

    def build_activation(self, facts: TargetFacts) -> Activation:
        return sigmoid_probabilities

    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter:
        return as_indicators

    @override
    def metric_kwargs(self, facts: TargetFacts) -> dict[str, Any]:
        return {"task": "binary"}


@objective_registry.register_instance(Objective.MULTILABEL)
class MultilabelObjective(TaskObjective):
    """Independent per-class probabilities: BCE per class, sigmoid for metrics."""

    needs_num_classes: ClassVar[bool] = True
    default_target_encoder: ClassVar[str | None] = "multilabel"

    def out_features(self, facts: TargetFacts) -> int:
        if facts.num_classes is None:
            raise LookupError("Multilabel needs num_classes; profile the data before building tasks.")
        return facts.num_classes

    def build_criterion(self, facts: TargetFacts) -> Criterion:
        return BinaryCrossEntropyCriterion()

    def build_activation(self, facts: TargetFacts) -> Activation:
        return sigmoid_probabilities

    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter:
        return as_indicators

    @override
    def metric_kwargs(self, facts: TargetFacts) -> dict[str, Any]:
        return {"task": "multilabel", "num_labels": facts.num_classes}


@objective_registry.register_instance(Objective.CONTINUOUS)
class ContinuousObjective(TaskObjective):
    """Real-valued targets, learned directly or through bins.

    Which of the two is in play is not a second declaration: it follows from the
    target encoder. A plain encoder yields one output against mean squared
    error; a binned one reports ``class_values``, and the same semantics is then
    learned as a distribution — cross-entropy over the bins plus a term on the
    expectation — and read back as the number it stands for. Either way the task
    is a regression and its metrics compare numbers.
    """

    default_target_encoder: ClassVar[str | None] = "scalar"

    def out_features(self, facts: TargetFacts) -> int:
        if facts.class_values is None:
            return 1
        return len(facts.class_values)

    def build_criterion(self, facts: TargetFacts) -> Criterion:
        if facts.class_values is None:
            return MeanSquaredErrorCriterion()
        # Cross-entropy leads: it shapes the whole distribution. The expectation term
        # is a correction that keeps the reported number aligned with the metric, and
        # is deliberately the lighter of the two — on its own, any distribution with
        # the right mean satisfies it. Config overrides both weights at once.
        return WeightedSumCriterion([(CrossEntropyCriterion(), 1.0), (ExpectationCriterion(facts.class_values), 0.5)])

    def build_activation(self, facts: TargetFacts) -> Activation:
        if facts.class_values is None:
            return squeeze_single_output
        return expectation_over(facts.class_values)

    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter:
        if facts.class_values is None:
            return float_for_loss
        return expectation_of(facts.class_values)


@objective_registry.register_instance(Objective.METRIC)
class MetricObjective(TaskObjective):
    """Embeddings shaped by comparison — against the batch, or against labels.

    The output already holds final embeddings, so the head is identity and ``out_features``
    does not apply. Supervision is usually the batch's structure (the in-batch diagonal); a
    task that declared a label column (ArcFace-style proxies) gets those labels delivered.
    """

    @override
    def out_features(self, facts: TargetFacts) -> None:
        """Nothing to project: the embedding the backbone produced is already the output."""
        return

    def build_criterion(self, facts: TargetFacts) -> Criterion:
        return InfoNceCriterion()

    def build_activation(self, facts: TargetFacts) -> Activation:
        return identity

    def build_target_adapter(self, facts: TargetFacts) -> TargetAdapter | None:
        return as_class_indices if facts.num_classes is not None else None
