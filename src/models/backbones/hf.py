"""Hugging Face backbone adapters — one class per modality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, override

from transformers import AutoConfig, AutoModel

from src.core.choices import one_of
from src.core.entities import Features
from src.core.ports import Backbone
from src.core.taxonomy import Modality, Stream
from src.models.registry import backbone_registry

if TYPE_CHECKING:
    from torch import Tensor

type Pooling = Literal["cls", "mean"]
"""How a sequence of token embeddings becomes one vector: the first token, or a mask-aware mean."""


@backbone_registry.register("hf_text")
class HFTextBackbone(Backbone):
    """A transformers text encoder as a pooled-embedding backbone.

    Reads token ids from ``inputs[input_name]`` and, when present, an
    attention mask from ``inputs[f"{input_name}_mask"]`` — the mask feeds
    both the model's attention and the mean pooling.

    Parameters:
        model_name (str): HF model id, e.g. ``"bert-base-uncased"``.
        pretrained (bool): Load weights; ``False`` builds the architecture
            from its config only (random init — tests, training from scratch).
        input_name (str): Which batch input holds the token ids.
        pooling (str): ``"cls"`` (first token) or ``"mean"`` (mask-aware).
        **kwargs: Forwarded verbatim to ``AutoModel.from_pretrained`` /
            ``AutoConfig.from_pretrained``.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        input_name: str = Modality.TEXT,
        pooling: Pooling = "cls",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        # Before the weights: a misspelt pooling is worth knowing about without
        # first downloading a model, and unchecked it would silently mean 'mean'.
        self._pooling = one_of(pooling, Pooling)
        self._architecture = model_name
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, **kwargs)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name, **kwargs))
        self._input_name = input_name
        self._feature_dim = int(self.model.config.hidden_size)

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        mask = inputs.get(f"{self._input_name}_mask")
        output = self.model(input_ids=inputs[self._input_name], attention_mask=mask)
        hidden: Tensor = output.last_hidden_state  # [B, L, D]
        if self._pooling == "cls":
            pooled = hidden[:, 0]
        elif mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)
        return Features(streams={Stream.FEATURES: pooled})

    @property
    @override
    def architecture(self) -> str:
        """The hub id, which is what a run is looked up by."""
        return self._architecture

    def feature_dim(self, stream: str) -> int:
        if stream != Stream.FEATURES:
            raise LookupError(f"HFTextBackbone exposes only the '{Stream.FEATURES}' stream, requested '{stream}'.")
        return self._feature_dim
