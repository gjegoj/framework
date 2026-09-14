"""One backbone over every draw of a sample: the views are folded into the batch, encoded, and unfolded."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor, nn

from src.core import Modality, TensorShape, TensorTree, require_tensor
from src.models.base import Backbone, required_input
from src.models.registry import backbone_registry

VIEW_AXIS = 1
"""Where a stack of draws sits before it is folded: after the samples they came from, before each draw.

One axis rather than a name in any declared shape. What an input *is* stays what one view is, which is
also what a deployed artifact takes, so views ride beside the batch axis — the one ``TensorShape``
deliberately does not carry either. Folding this one into the batch is what leaves the draws of a
sample adjacent, which is the order the objective reading them relies on.
"""


@backbone_registry.register("multiview")
class MultiViewBackbone(Backbone):
    """Encodes every view with one set of weights, by folding the views into the batch to do it.

    Folded rather than looped: the draws of one sample are independent, so putting them in the batch is
    the same arithmetic in one pass, and it is what makes the weights shared rather than merely equal.

    Folded and left that way. Unfolding here would give this one kind of run an output a rank deeper
    than every other — what a task declares it produces is what *one* sample's answer is, and a run
    that draws views has changed how many samples a batch holds rather than what each answer is. So the
    views go into the batch and stay there, the heads are built and applied exactly as they always are,
    and the objective that compares draws recovers them from the one thing that still counts samples:
    how many targets it was handed against how many rows it was given. The order this leaves them in is
    that contract — every draw of a sample before the next sample's — and the objective names it too.

    How many views there are is therefore declared in one place only, on the transform that draws them.
    Nothing here needs the number, and a run whose chain draws none is refused rather than mangled: the
    axis this would fold is then the channels, and what the wrapped family is handed has lost a
    dimension, which every one of them refuses by name.

    Declared around a backbone rather than instead of one, so that every family this framework has stays
    available to a run that draws views, and none of them learns what a view is.

    Parameters:
        backbone: The network that encodes one view; any family, declared by ``_target_``.
        input_name: Which of the batch's inputs arrives as a stack of views.
    """

    def __init__(self, backbone: Backbone, input_name: str = Modality.IMAGE) -> None:
        super().__init__()
        if not isinstance(backbone, Backbone):
            raise TypeError(
                f"{type(backbone).__name__} is not a Backbone: this one draws views through another and "
                "publishes what that one publishes, so it needs a network that names its feature streams."
            )
        self.backbone = backbone
        self.input_name = input_name

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        """What the wrapped backbone publishes, unchanged: a view encodes to exactly what a picture does."""
        return self.backbone.feature_shapes

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        drawn = require_tensor(required_input(inputs, self.input_name, type(self).__name__), name=self.input_name)
        return cast("Mapping[str, Tensor]", self.backbone({**inputs, self.input_name: drawn.flatten(0, VIEW_AXIS)}))

    def native_head(self, stream: str, out_features: int) -> nn.Module | None:
        """Whatever the wrapped family brings: drawing views changes nothing about what a head is."""
        return self.backbone.native_head(stream, out_features)
