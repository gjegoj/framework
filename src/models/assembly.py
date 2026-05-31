"""Model assembly: a backbone plus per-task heads, wired into one nn.Module.

``CompositeModel`` owns a single backbone and an ``nn.ModuleDict`` of heads
(keyed by task name), routing the declared feature stream to each head. Heads
are *derived from tasks*: ``build_composite_model`` sizes each head from the
backbone's feature dimension and constructs it via the head registry.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.core.entities import HeadSpec, ModelOutput
from src.core.ports import Backbone, Head
from src.models.registry import head_builders


class CompositeModel(nn.Module):
    """A shared backbone with one head per task.

    Parameters:
        backbone (Backbone): Feature extractor producing a ``FeatureBundle``.
        heads (dict[str, Head]): Per-task heads keyed by task name.
        feature_keys (dict[str, str]): Per-task feature stream each head consumes.
    """

    def __init__(self, backbone: Backbone, heads: dict[str, Head], feature_keys: dict[str, str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self._feature_keys = feature_keys

    def forward(self, inputs: dict[str, Tensor]) -> ModelOutput:
        features = self.backbone(inputs)
        task_logits = {name: self.heads[name](features[self._feature_keys[name]]) for name in self.heads}
        return ModelOutput(features=features, task_logits=task_logits)


def build_composite_model(backbone: Backbone, specs: dict[str, HeadSpec]) -> CompositeModel:
    """Assemble a ``CompositeModel`` from a backbone and per-task head specs.

    Parameters:
        backbone (Backbone): The shared feature extractor.
        specs (dict[str, HeadSpec]): Head spec per task name (out-features resolved).

    Returns:
        CompositeModel: The assembled multi-head model.
    """
    heads: dict[str, Head] = {}
    feature_keys: dict[str, str] = {}
    for name, spec in specs.items():
        in_features = backbone.feature_dim(spec.feature_key)
        heads[name] = head_builders.create(
            spec.kind,
            in_features=in_features,
            out_features=spec.out_features,
            **spec.options,
        )
        feature_keys[name] = spec.feature_key
    return CompositeModel(backbone=backbone, heads=heads, feature_keys=feature_keys)
