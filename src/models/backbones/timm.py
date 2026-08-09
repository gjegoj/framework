"""timm backbone adapter: any timm model as a pooled-feature encoder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import timm
from torch import nn

from src.core.entities import Features
from src.core.ports import Backbone
from src.core.taxonomy import Modality, Stream
from src.models.backbones.checkpoints import (
    classifier_prefixes,
    load_arrived_weights,
    transplanted_classifier,
)
from src.models.registry import backbone_registry

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from torch import Tensor


@backbone_registry.register("timm")
class TimmBackbone(Backbone):
    """Wraps any timm model as a backbone with one pooled feature stream.

    The model is created with ``num_classes=0``, so its forward returns the
    pooled feature vector ``[B, num_features]``, exposed under
    ``Stream.FEATURES``.

    Parameters:
        model_name (str): timm model id, e.g. ``"resnet18"`` (``name`` is
            taken by the registry key in component configs).
        pretrained (bool): Load pretrained weights (needs network or cache).
            Moot when ``checkpoint_path`` is given: a file is the weight
            source, and downloading the hub to overwrite it is dead traffic.
        input_name (str): Which batch input to encode.
        checkpoint_path (str | Path | None): Arrived weights of this
            architecture — a full timm model's checkpoint. The backbone
            tensors load; the classifier is stashed for ``native_head`` to
            transplant. ``use_ema``/``strict``/``weights_only`` carry
            ``timm.load_checkpoint``'s own names and defaults — mechanics in
            the ``checkpoints`` package.
        use_ema (bool): Prefer the checkpoint's EMA branch when present.
        strict (bool): Refuse a checkpoint that does not match the backbone
            exactly; ``False`` allows deliberate partial loads, reported
            loudly and refused entirely when nothing matched.
        weights_only (bool): ``torch.load`` safety knob, forwarded.
        **kwargs: Forwarded verbatim to ``timm.create_model``
            (``in_chans``, ``drop_rate``, ...).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        input_name: str = Modality.IMAGE,
        checkpoint_path: str | Path | None = None,
        use_ema: bool = True,
        strict: bool = True,
        weights_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained and checkpoint_path is None, num_classes=0, **kwargs
        )
        self._feature_dim = cast(int, self.model.num_features)
        self._input_name = input_name
        self._carried_classifier: dict[str, Tensor] | None = None
        if checkpoint_path is not None:
            self._carried_classifier = load_arrived_weights(
                self.model,
                model_name,
                checkpoint_path,
                use_ema=use_ema,
                strict=strict,
                weights_only=weights_only,
                stash_prefixes=classifier_prefixes(self.model),
            )

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        return Features(streams={Stream.FEATURES: self.model(inputs[self._input_name])})

    @property
    @override
    def architecture(self) -> str:
        """timm's own name for what it built: measured, it drops a weights tag (`resnet18.a1_in1k`).

        Falls back to the port's default rather than raising: this names a run in a
        tracker, and a model whose config is shaped unusually should not stop one.
        """
        declared = cast("dict[str, Any]", getattr(self.model, "default_cfg", {}))
        return str(declared.get("architecture") or type(self).__name__)

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: self._feature_dim}

    @override
    def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
        if stream != Stream.FEATURES:
            return None
        if self._carried_classifier is not None:
            return transplanted_classifier(self._carried_classifier, in_features, out_features)
        from timm.layers import create_classifier

        _, classifier = create_classifier(in_features, out_features)
        return cast("nn.Module", classifier)
