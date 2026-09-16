"""What a label means, independent of where it sits: one class per position, one score, or one score per label.

Each mixin answers the three questions a step asks of a target and the one a head asks of the vocabulary.
Paired with a topology (a whole sample, or every pixel of one) they make the kinds a run declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, override

from torch import Tensor
from torch.nn.functional import one_hot

from src.core import FEATURE_AXIS, Batch, Representation, Semantics, TensorTree, drop_feature_axis
from src.tasks.base import LossDeclaration, TargetEncoderDeclaration, Task

DECISION = 0.5
"""Where a score becomes a decision: at or above this, the label holds.

One home for its two readers. A task hardens its own target here, and a page reads a prediction the
same way — a page allowed a line of its own would be showing mistakes the run's metrics do not count.
It is also torchmetrics' own default, so a run that declares no threshold has all three agreeing.
"""

CLASSIFICATION_METRICS: Mapping[str, Mapping[str, object]] = {
    # `average="none"` asks for the per-class vector; a binary metric ignores it and answers with the
    # positive class alone, so one table serves every label semantics.
    "f1": {"name": "f1", "average": "none"},
    "precision": {"name": "precision", "average": "none"},
    "recall": {"name": "recall", "average": "none"},
    "confusion_matrix": {"name": "confusion_matrix", "normalize": "true"},
}


class MulticlassSemantics(Task):
    """One class per position, chosen from a declared vocabulary; the model scores every class."""

    semantics: ClassVar[Semantics | None] = Semantics.MULTICLASS
    publishes: ClassVar[Representation] = Representation.PROBABILITIES
    default_target_encoder: ClassVar[TargetEncoderDeclaration | None] = "label"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = CLASSIFICATION_METRICS

    @property
    def default_loss(self) -> LossDeclaration:
        return "cross_entropy"

    def out_features(self) -> int:
        if self.info.num_classes is None or self.info.num_classes < 2:
            raise ValueError(
                f"{type(self).__name__} chooses between classes, so it needs at least two; the target "
                f"declares {self.info.num_classes if self.info.num_classes is not None else 'none'}."
            )
        return self.info.num_classes

    def loss_target(self, batch: Batch) -> Tensor:
        """An index as it stands; a share of each class, once a batch transform mixed two samples, as it stands too."""
        target = self.target(batch)
        return target if target.is_floating_point() else target.long()

    def metric_view(self, batch: Batch) -> Tensor:
        """The class the sample mostly is: a metric ranks against one class, however soft the target became."""
        target = self.target(batch)
        return target.argmax(dim=FEATURE_AXIS) if target.is_floating_point() else target.long()

    @override
    def soften(self, target: Tensor) -> Tensor:
        """An index widened into the share of each class it stands for: all of one, none of the rest."""
        if target.is_floating_point():
            return target
        return one_hot(target.long(), num_classes=self.out_features()).movedim(-1, FEATURE_AXIS).float()

    @override
    def publish(self, projected: Tensor) -> TensorTree:
        """A share of every class, over the axis the head projected along."""
        return projected.softmax(dim=FEATURE_AXIS)


class BinarySemantics(Task):
    """One score per position: whether the thing is there. No vocabulary, one output."""

    semantics: ClassVar[Semantics | None] = Semantics.BINARY
    publishes: ClassVar[Representation] = Representation.PROBABILITIES
    default_target_encoder: ClassVar[TargetEncoderDeclaration | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = CLASSIFICATION_METRICS

    @property
    def default_loss(self) -> LossDeclaration:
        return "bce"

    def out_features(self) -> int:
        return 1

    def loss_target(self, batch: Batch) -> Tensor:
        """The share of the label the sample carries — one for a plain target, a fraction after a mix."""
        return self.target(batch).float()

    def metric_view(self, batch: Batch) -> Tensor:
        return (self.target(batch) >= DECISION).long()

    @override
    def publish(self, projected: Tensor) -> TensorTree:
        """One score, without the axis it was projected along: there is only ever the one."""
        return drop_feature_axis(projected).sigmoid()


class MultilabelSemantics(Task):
    """Any number of labels per sample: one independent score for each of the declared classes."""

    semantics: ClassVar[Semantics | None] = Semantics.MULTILABEL
    publishes: ClassVar[Representation] = Representation.PROBABILITIES
    default_target_encoder: ClassVar[TargetEncoderDeclaration | None] = "multilabel"
    # A multilabel confusion matrix is one small matrix per label, which reports as nothing useful.
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {
        label: declared for label, declared in CLASSIFICATION_METRICS.items() if label != "confusion_matrix"
    }

    @property
    def default_loss(self) -> LossDeclaration:
        return "bce"

    def out_features(self) -> int:
        if self.info.num_classes is None:
            raise ValueError(f"{type(self).__name__} scores one output per label, so its classes must be declared.")
        return self.info.num_classes

    def loss_target(self, batch: Batch) -> Tensor:
        return self.target(batch).float()

    def metric_view(self, batch: Batch) -> Tensor:
        return (self.target(batch) >= DECISION).long()

    @override
    def publish(self, projected: Tensor) -> TensorTree:
        """A score per label, each answering about its own label and none of them about the rest."""
        return projected.sigmoid()
