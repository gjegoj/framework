"""ONNX/TorchScript-friendly wrappers over ``CompositeModel``.

Each wrapper exposes a plain tensor-in / tensor-out ``forward`` so export backends
can trace a stable graph without dicts or dataclasses.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from src.core.keys import IMAGE
from src.core.ports import Activation, Backbone, Head
from src.models.assembly import CompositeModel


class CombinedExportModel(nn.Module):
    """Single image in → per-task predictions out (tuple, fixed task order).

    Each output is ``task.activation(head_logits)`` — ready for deployment inference,
    matching the post-activation preds used in metrics and visualization.

    Parameters:
        model (CompositeModel): Assembled backbone + heads.
        task_names (tuple[str, ...]): Task names defining output order.
        activations (dict[str, Activation]): Per-task logits→predictions maps.
        input_key (str): ``Batch.inputs`` alias fed to the backbone.
    """

    def __init__(
        self,
        model: CompositeModel,
        task_names: tuple[str, ...],
        activations: dict[str, Activation],
        input_key: str = IMAGE,
    ) -> None:
        super().__init__()
        self._model = model
        self._task_names = task_names
        self._activations = activations
        self._input_key = input_key

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        output = self._model({self._input_key: image})
        return tuple(self._activations[name](output.task_logits[name]) for name in self._task_names)


class BackboneExportModel(nn.Module):
    """Image in → named backbone feature streams out (tuple, fixed stream order).

    Parameters:
        backbone (Backbone): Feature extractor.
        stream_keys (tuple[str, ...]): ``FeatureBundle`` keys to export.
        input_key (str): Input alias passed to the backbone.
    """

    def __init__(self, backbone: Backbone, stream_keys: tuple[str, ...], input_key: str = IMAGE) -> None:
        super().__init__()
        self._backbone = backbone
        self._stream_keys = stream_keys
        self._input_key = input_key

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        bundle = self._backbone({self._input_key: image})
        return tuple(bundle[key] for key in self._stream_keys)


class HeadExportModel(nn.Module):
    """Feature tensor in → task predictions out (head + activation, no backbone).

    Parameters:
        head (Head): Task head module.
        activation (Activation): Maps head logits to deployment predictions.
    """

    def __init__(self, head: Head, activation: Activation) -> None:
        super().__init__()
        self._head = head
        self._activation = activation

    def forward(self, features: Tensor) -> Tensor:
        return self._activation(cast(Tensor, self._head(features)))
