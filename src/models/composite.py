"""The composite family: one backbone encodes, one head per task reads the stream it declared."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor, nn

from src.core import ModelOutput, TensorTree, validate_name
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
        if not heads:
            raise ValueError("A composite model serves at least one task; none was declared.")
        published = backbone.feature_shapes
        for name, connection in heads.items():
            validate_name(name, kind="Task")
            if connection.input not in published:
                raise ValueError(
                    f"Head {name!r} reads {connection.input!r}, but {type(backbone).__name__} publishes "
                    f"{', '.join(published)}."
                )
        self.backbone = backbone
        self.heads = nn.ModuleDict({name: connection.head for name, connection in heads.items()})
        self._inputs = {name: connection.input for name, connection in heads.items()}

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = self.backbone(inputs)
        outputs = {name: cast(Tensor, self.heads[name](features[stream])) for name, stream in self._inputs.items()}
        return ModelOutput(outputs=outputs, features=features)
