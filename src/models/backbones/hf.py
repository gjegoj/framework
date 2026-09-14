"""Transformers as a backbone: a family's own encoder, read down to the one vector a head is sized by."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from torch import Tensor

from src.core import Axis, Modality, Stream, TensorShape, TensorTree, require_named_tensors
from src.models.base import Backbone, required_input
from src.models.registry import backbone_registry

TOKEN_AXIS = 1
"""Where a family's sequence output counts its tokens: ``[batch, tokens, width]``."""

MASK = "attention_mask"
"""What every family of this library calls the tensor saying which positions are words and which are padding."""

POOLINGS = ("cls", "mean")
"""How a sequence becomes one vector. Both are real choices, so both are declared rather than assumed:
``cls`` is what a family pretrained with a summary marker put first, and ``mean`` is what reads every
word of a sentence the family was not pretrained to summarise."""


@backbone_registry.register("hf_text")
class HFTextBackbone(Backbone):
    """Encodes a caption to one ``[batch, width]`` stream, whichever family the declaration names.

    The input arrives as the tree its own tokenizer made — ids, the mask, and whatever else that family
    reads — and this hands the family back exactly that, under the names it uses. A convention of its
    own (``<input>_mask`` beside the input) would be a second place where "which tokens are words" is
    written, owned by neither the encoder that knows it nor the model that needs it.

    No ``**options``: a knob of ``from_pretrained`` arrives named in this signature together with the
    run that declares it. Passing a declaration through untouched is what let ``checkpoint_path`` reach
    timm and fail with a word about ``fc.weight``, naming neither the declaration nor the fix.

    Args:
        model_name: A hub id or a directory. The encoder that tokenizes this input reads the same name,
            and a run declares it once: two spellings of one family is a vocabulary mismatch, which
            shows up as nothing but a worse number.
        pooling: ``cls`` or ``mean``; see ``POOLINGS``.
        input_name: Which of the batch's inputs to read.
    """

    def __init__(self, model_name: str, pooling: str = "mean", input_name: str = Modality.TEXT) -> None:
        # Imported here, not at module scope: this package's facade imports every backbone so that a
        # name resolves, and `transformers` costs a measured 6.1 s — a run over pixels would pay it.
        from transformers import AutoModel

        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling is one of {', '.join(POOLINGS)}, got {pooling!r}.")
        self.model = AutoModel.from_pretrained(model_name)
        self.input_name = input_name
        self.pooling = pooling
        self.width = int(self.model.config.hidden_size)

    @property
    def feature_shapes(self) -> Mapping[str, TensorShape]:
        return {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(self.width,))}

    def forward(self, inputs: Mapping[str, TensorTree]) -> Mapping[str, Tensor]:
        tokens = require_named_tensors(
            required_input(inputs, self.input_name, type(self).__name__), name=self.input_name
        )
        hidden = cast("Any", self.model(**tokens)).last_hidden_state
        return {Stream.POOLED: self._pooled(hidden, tokens)}

    def _pooled(self, hidden: Tensor, tokens: Mapping[str, Tensor]) -> Tensor:
        """One vector per caption, with the padding that made a batch of them left out of the answer.

        Measured on a two-layer family: a mean over every position, padding included, moves the same
        sentence by more than a unit and keeps moving it as the declared length grows — the caption
        would mean one thing in a run with room for long ones and another in a run without.
        """
        if self.pooling == "cls":
            return hidden[:, 0]
        if MASK not in tokens:
            carried = ", ".join(sorted(tokens)) or "nothing"
            raise LookupError(
                f"Pooling a caption by its mean needs {MASK!r} to say which positions are words, and "
                f"{self.input_name!r} carries {carried}. A family's own tokenizer publishes it; an encoder "
                f"that does not can be read with pooling: cls, which takes one position and counts nothing."
            )
        words = tokens[MASK].unsqueeze(-1).to(hidden.dtype)
        return (hidden * words).sum(dim=TOKEN_AXIS) / words.sum(dim=TOKEN_AXIS).clamp(min=1.0)
