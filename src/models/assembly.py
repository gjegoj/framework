"""Model assembly: a backbone plus per-task heads, wired into one nn.Module.

``CompositeModel`` owns a single backbone and an ``nn.ModuleDict`` of heads
(keyed by task name), routing the declared feature stream to each head. Heads
are *derived from tasks*: ``build_composite_model`` sizes each head from the
backbone's feature dimension and constructs it via one of three paths:

1. **``_target_``** — fully custom head instantiated from an import path.
2. **Native** — backbone provides its own architecture-appropriate head.
3. **Registry** — ``head_builders.create(kind, in_features, out_features)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from src.core.entities import FeatureBundle, HeadSpec, ModelOutput
from src.core.instantiate import instantiate
from src.core.keys import IMAGE
from src.core.ports import Backbone, Head
from src.models.heads import AsHead
from src.models.registry import head_builders


class _FeatureExtractor(ABC):
    """Strategy producing one task's head input from the backbone.

    Each concrete extractor turns the raw inputs into the tensor its head
    consumes — reading a stream from the shared backbone output, stacking N views
    through one shared backbone, or stacking N separate-encoder streams — while
    reusing (and populating) the shared ``FeatureBundle`` so sibling tasks in one
    forward pass don't re-run the backbone.
    """

    @abstractmethod
    def extract(
        self,
        backbone: Backbone,
        inputs: dict[str, Tensor],
        shared_features: FeatureBundle | None,
    ) -> tuple[Tensor, FeatureBundle | None]:
        """Return ``(head_input, shared_features)`` — the latter cached for reuse."""

    @staticmethod
    def _ensure_shared_features(
        backbone: Backbone, inputs: dict[str, Tensor], shared_features: FeatureBundle | None
    ) -> FeatureBundle:
        """Run the backbone once, or reuse the bundle already cached this forward pass."""
        return shared_features if shared_features is not None else backbone(inputs)


@dataclass(frozen=True)
class _SingleViewExtractor(_FeatureExtractor):
    """Extracts features for a standard single-image task.

    Runs the backbone once on the full inputs dict and caches the resulting
    ``FeatureBundle`` so sibling tasks in the same forward pass reuse it.
    """

    feature_key: str

    def extract(
        self,
        backbone: Backbone,
        inputs: dict[str, Tensor],
        shared_features: FeatureBundle | None,
    ) -> tuple[Tensor, FeatureBundle | None]:
        shared_features = self._ensure_shared_features(backbone, inputs, shared_features)
        return shared_features[self.feature_key], shared_features


@dataclass(frozen=True)
class _MultiViewExtractor(_FeatureExtractor):
    """Extracts features for a multi-view (Siamese) ranking task.

    Stacks all N view tensors into one ``[B*N, ...]`` batch, runs the shared
    backbone once, then reshapes to ``[B, N, D]``.  Backbone weights are
    shared across all views — one forward pass regardless of N.
    """

    feature_key: str
    view_keys: tuple[str, ...]

    def extract(
        self,
        backbone: Backbone,
        inputs: dict[str, Tensor],
        shared_features: FeatureBundle | None,
    ) -> tuple[Tensor, FeatureBundle | None]:
        stacked_views = torch.cat([inputs[key] for key in self.view_keys], dim=0)
        view_features = backbone({IMAGE: stacked_views})
        embeddings = view_features[self.feature_key]
        batch_size = inputs[self.view_keys[0]].size(0)
        num_views = len(self.view_keys)
        reshaped_embeddings = embeddings.view(num_views, batch_size, -1).permute(1, 0, 2)
        return reshaped_embeddings, shared_features


@dataclass(frozen=True)
class _MultiStreamExtractor(_FeatureExtractor):
    """Extracts features for a multi-encoder (dual/multi-stream) contrastive task.

    Reads N named streams produced by a ``MultiEncoderBackbone`` (one per
    encoder, separate weights) and stacks them into ``[B, N, D]``.  Unlike
    ``_MultiViewExtractor`` it does not re-run the backbone — the streams are
    already in the shared ``FeatureBundle``.
    """

    stream_keys: tuple[str, ...]

    def extract(
        self,
        backbone: Backbone,
        inputs: dict[str, Tensor],
        shared_features: FeatureBundle | None,
    ) -> tuple[Tensor, FeatureBundle | None]:
        shared_features = self._ensure_shared_features(backbone, inputs, shared_features)
        embeddings = [shared_features[key] for key in self.stream_keys]
        return torch.stack(embeddings, dim=1), shared_features


class CompositeModel(nn.Module):
    """A shared backbone with one head per task.

    Each task has a ``_FeatureExtractor`` that knows how to produce features
    for it — either by reading a stream from the shared backbone output
    (single-view) or by stacking N views and running the backbone once
    (multi-view / Siamese).  The ``forward`` loop is uniform: no branching.

    Parameters:
        backbone (Backbone): Feature extractor producing a ``FeatureBundle``.
        heads (dict[str, Head]): Per-task heads keyed by task name.
        extractors (dict[str, _FeatureExtractor]): Per-task feature extractor.
    """

    def __init__(
        self,
        backbone: Backbone,
        heads: dict[str, Head],
        extractors: dict[str, _FeatureExtractor],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(heads)
        self._extractors = extractors

    def forward(self, inputs: dict[str, Tensor]) -> ModelOutput:
        shared_features: FeatureBundle | None = None
        task_logits: dict[str, Tensor] = {}

        for task_name in self.heads:
            features, shared_features = self._extractors[task_name].extract(self.backbone, inputs, shared_features)
            task_logits[task_name] = self.heads[task_name](features)

        return ModelOutput(
            features=shared_features if shared_features is not None else FeatureBundle(streams={}),
            task_logits=task_logits,
        )

    def __call__(self, inputs: dict[str, Tensor]) -> ModelOutput:
        # Typed delegate to nn.Module.__call__ so callers get ModelOutput, not Any
        # (torch types Module.__call__ as Callable[..., Any]); hooks still run.
        return cast(ModelOutput, super().__call__(inputs))


def build_composite_model(backbone: Backbone, specs: dict[str, HeadSpec]) -> CompositeModel:
    """Assemble a ``CompositeModel`` from a backbone and per-task head specs.

    Each task gets a feature extractor chosen from its ``HeadSpec``:
    ``stream_keys`` → multi-encoder (contrastive), ``view_keys`` → multi-view
    (Siamese ranking), neither → single-view.

    Parameters:
        backbone (Backbone): The shared feature extractor.
        specs (dict[str, HeadSpec]): Head spec per task name (out-features resolved).

    Returns:
        CompositeModel: The assembled multi-head model.
    """
    heads: dict[str, Head] = {}
    extractors: dict[str, _FeatureExtractor] = {}

    for name, spec in specs.items():
        extractor, in_features = _build_extractor(spec, backbone)
        heads[name] = _build_head(backbone, spec, in_features)
        extractors[name] = extractor

    return CompositeModel(backbone=backbone, heads=heads, extractors=extractors)


def _build_extractor(spec: HeadSpec, backbone: Backbone) -> tuple[_FeatureExtractor, int]:
    """Select the feature extractor for one task and the in-features to size its head.

    The in-features come from the stream the extractor actually reads, so the
    head is sized against the right dimension in every mode.
    """
    if spec.stream_keys is not None:
        in_features = backbone.feature_dim(spec.stream_keys[0])
        return _MultiStreamExtractor(stream_keys=spec.stream_keys), in_features
    if spec.view_keys is not None:
        in_features = backbone.feature_dim(spec.feature_key)
        return _MultiViewExtractor(feature_key=spec.feature_key, view_keys=spec.view_keys), in_features
    return _SingleViewExtractor(feature_key=spec.feature_key), backbone.feature_dim(spec.feature_key)


def _build_head(backbone: Backbone, spec: HeadSpec, in_features: int) -> Head:
    """Construct one task head following the three-mode priority order."""

    # 1. _target_ mode: user owns the head class; in_features / in_channels injected.
    if spec.target is not None:
        raw = instantiate(
            {"_target_": spec.target, **spec.options},
            in_features=in_features,
            in_channels=in_features,
            out_features=spec.out_features,
            out_channels=spec.out_features,
        )
        return AsHead(raw)

    # 2. Native mode: ask the backbone for its architecture-appropriate head.
    if spec.prefer_native:
        raw = backbone.native_head(spec.feature_key, in_features, spec.out_features)
        if raw is not None:
            return AsHead(raw)
        # Backbone returned None — fall through to the registry.

    # 3. Registry mode.
    return head_builders.create(spec.kind, in_features=in_features, out_features=spec.out_features, **spec.options)
