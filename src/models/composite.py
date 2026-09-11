"""The composite family: one backbone encodes, one head per task reads the stream it declared."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from torch import Tensor, nn

from src.core import ModelOutput, TensorTree
from src.models.base import Backbone, HeadConnection, Model
from src.models.registry import model_registry


@model_registry.register("composite")
class CompositeModel(Model):
    """Encode once, serve every task from a named stream.

    Heads register as ``heads.<task>``, the backbone as ``backbone``: those paths are the contract a
    freeze callback, a checkpoint and a parameter group address, so they are part of the design.
    """

    def __init__(self, backbone: Backbone, heads: Mapping[str, HeadConnection]) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({name: connection.head for name, connection in heads.items()})
        self._streams = {name: connection.stream for name, connection in heads.items()}

    def parameters_of(self, task: str) -> Iterable[nn.Parameter]:
        """A composite gives each task its head and shares the backbone; the split is exactly that."""
        return self.heads[task].parameters() if task in self.heads else ()

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = self.backbone(inputs)
        outputs = {name: cast(Tensor, self.heads[name](features[stream])) for name, stream in self._streams.items()}
        return ModelOutput(outputs=outputs, features=features)
