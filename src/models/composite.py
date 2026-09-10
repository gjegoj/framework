"""One graph for a ready backbone and heads; no models, tasks or configs are constructed here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from torch import nn

from src.core import ModelOutput, ShapeTree, TensorTree
from src.core.entities import validate_name
from src.models.backbones.base import Backbone
from src.models.base import Model


@dataclass(frozen=True, slots=True)
class HeadConnection:
    """One ready head and its feature selection; the mapping key names its output."""

    head: nn.Module
    input: str | tuple[str, ...]

    def __post_init__(self) -> None:
        names = (self.input,) if isinstance(self.input, str) else self.input
        if not names or any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise ValueError("Head inputs require distinct, nonblank feature names.")


def select_features(features: Mapping[str, TensorTree], selection: str | tuple[str, ...]) -> TensorTree:
    """Select one value or an ordered mapping; adapters and composed models use the same rule."""
    return features[selection] if isinstance(selection, str) else {name: features[name] for name in selection}


class CompositeModel(Model):
    """Register a ready graph once; task semantics and dimension inference belong to assembly."""

    def __init__(self, backbone: Backbone, heads: Mapping[str, HeadConnection]) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self._connections: dict[str, tuple[str, str | tuple[str, ...]]] = {}
        registered: dict[int, str] = {}
        for name, connection in heads.items():
            validate_name(name)
            names = (connection.input,) if isinstance(connection.input, str) else connection.input
            missing = set(names) - backbone.feature_shapes.keys()
            if missing:
                raise ValueError(f"Head {name!r} requests unavailable features: {sorted(missing)}.")
            if id(connection.head) not in registered:
                self.heads[name] = connection.head
                registered[id(connection.head)] = name
            self._connections[name] = (registered[id(connection.head)], connection.input)

    @property
    def feature_shapes(self) -> Mapping[str, ShapeTree]:
        return self.backbone.feature_shapes

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        features = cast(Mapping[str, TensorTree], self.backbone(inputs))
        outputs = {
            name: cast(TensorTree, self.heads[head_name](select_features(features, selection)))
            for name, (head_name, selection) in self._connections.items()
        }
        return ModelOutput(outputs=outputs, features=features)
