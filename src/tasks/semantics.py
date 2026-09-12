"""What a label means, independent of where it sits: one class per position, one score, or one score per label.

Each mixin answers the three questions a step asks of a target and the one a head asks of the vocabulary.
Paired with a topology (a whole sample, or every pixel of one) they make the kinds a run declares.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, override

from torch import Tensor
from torch.nn.functional import one_hot

from src.core import CLASS_AXIS, Batch, ModelOutput, Semantics, TargetInfo, TensorTree, drop_class_axis
from src.tasks.base import LossDeclaration, Task

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
    default_target_encoder: ClassVar[str | None] = "label"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = CLASSIFICATION_METRICS

    @property
    def default_loss(self) -> LossDeclaration:
        return "cross_entropy"

    @classmethod
    def out_features(cls, info: TargetInfo) -> int:
        if info.num_classes is None or info.num_classes < 2:
            raise ValueError(
                f"{cls.__name__} chooses between classes, so it needs at least two; the target declares "
                f"{info.num_classes if info.num_classes is not None else 'none'}."
            )
        return info.num_classes

    def loss_target(self, batch: Batch) -> Tensor:
        """An index as it stands; a share of each class, once a batch transform mixed two samples, as it stands too."""
        target = self.target(batch)
        return target if target.is_floating_point() else target.long()

    def metric_view(self, batch: Batch) -> Tensor:
        """The class the sample mostly is: a metric ranks against one class, however soft the target became."""
        target = self.target(batch)
        return target.argmax(dim=CLASS_AXIS) if target.is_floating_point() else target.long()

    @override
    def soften(self, target: Tensor) -> Tensor:
        """An index widened into the share of each class it stands for: all of one, none of the rest."""
        if target.is_floating_point():
            return target
        return one_hot(target.long(), num_classes=self.out_features(self.info)).movedim(-1, CLASS_AXIS).float()

    def postprocess(self, output: ModelOutput) -> TensorTree:
        return self.raw(output).softmax(dim=CLASS_AXIS)


class BinarySemantics(Task):
    """One score per position: whether the thing is there. No vocabulary, one output."""

    semantics: ClassVar[Semantics | None] = Semantics.BINARY
    default_target_encoder: ClassVar[str | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = CLASSIFICATION_METRICS

    @property
    def default_loss(self) -> LossDeclaration:
        return "bce"

    @classmethod
    def out_features(cls, info: TargetInfo) -> int:
        return 1

    def loss_target(self, batch: Batch) -> Tensor:
        """The share of the label the sample carries — one for a plain target, a fraction after a mix."""
        return self.target(batch).float()

    def metric_view(self, batch: Batch) -> Tensor:
        return (self.target(batch) >= 0.5).long()

    def postprocess(self, output: ModelOutput) -> TensorTree:
        return drop_class_axis(self.raw(output)).sigmoid()


class MultilabelSemantics(Task):
    """Any number of labels per sample: one independent score for each of the declared classes."""

    semantics: ClassVar[Semantics | None] = Semantics.MULTILABEL
    default_target_encoder: ClassVar[str | None] = "multilabel"
    # A multilabel confusion matrix is one small matrix per label, which reports as nothing useful.
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {
        label: declared for label, declared in CLASSIFICATION_METRICS.items() if label != "confusion_matrix"
    }

    @property
    def default_loss(self) -> LossDeclaration:
        return "bce"

    @classmethod
    def out_features(cls, info: TargetInfo) -> int:
        if info.num_classes is None:
            raise ValueError(f"{cls.__name__} scores one output per label, so its classes must be declared.")
        return info.num_classes

    def loss_target(self, batch: Batch) -> Tensor:
        return self.target(batch).float()

    def metric_view(self, batch: Batch) -> Tensor:
        return (self.target(batch) >= 0.5).long()

    def postprocess(self, output: ModelOutput) -> TensorTree:
        return self.raw(output).sigmoid()
