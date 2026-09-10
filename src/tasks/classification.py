"""One mutually exclusive class per sample; the model supplies unnormalized scores."""

from collections.abc import Mapping
from typing import ClassVar, cast

from torch import Tensor

from src.core import Axis, Batch, ModelOutput, Stream, TargetInfo, TensorShape
from src.core.types import require_tensor
from src.tasks.base import Task


class ClassificationTask(Task):
    default_head: ClassVar[Mapping[str, object]] = {"name": "linear", "input": Stream.POOLED}
    default_loss = "cross_entropy"
    default_target_encoder = "label"

    @classmethod
    def output_shape(cls, target_info: TargetInfo) -> TensorShape:
        if target_info.num_classes is None or target_info.num_classes < 2:
            raise ValueError("Multiclass classification requires at least two declared classes.")
        return TensorShape(axes=(Axis.CLASSES,), sizes=(target_info.num_classes,))

    def postprocess(self, model_output: ModelOutput, batch: Batch) -> Tensor:
        return require_tensor(self.select_output(model_output), name=self.name).softmax(dim=-1)

    def prepare_metric_targets(self, batch: Batch) -> Tensor | None:
        targets = cast(Tensor | None, batch.targets.get(self.name))
        # Mixup targets remain soft for the loss; metrics compare class indices.
        return targets.argmax(dim=-1) if targets is not None and targets.ndim == 2 else targets
