"""Text as the tree of tensors its family reads: one place where a sentence becomes numbers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.core import Axis, InputInfo, Modality, TensorShape
from src.data.base import InputEncoder
from src.data.registry import input_encoder_registry

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@input_encoder_registry.register("text")
class TextEncoder(InputEncoder):
    """Declares how a caption reaches the model — whose tokenizer reads it, and at what length.

    What a text input *is* is a tree rather than one tensor: a family reads ids beside a mask, and some
    read more than that. Which names those are is asked of the tokenizer (``model_input_names``) rather
    than listed here, so this encoder serves a family whose inputs it was never told about, and what it
    publishes cannot drift from what it produces — both read the same tuple.

    The mask is part of that tree for the same reason. A convention naming it after the input
    (``<input>_mask``) would put the fact that a sentence was padded outside the value it belongs to,
    owned by nobody and re-derived by every reader.

    Args:
        model_name: A hub id or a directory — whatever ``from_pretrained`` takes. The backbone over
            this input reads the same name; a run declares it once and interpolates it, because two
            spellings of one family is a vocabulary mismatch that only shows as a worse number.
        max_length: How many tokens every caption becomes, shorter ones padded and longer ones cut.
            Fixed rather than per-batch so that a batch stacks, which is what ``StackCollator`` does
            to every leaf of this tree.
    """

    def __init__(self, model_name: str, max_length: int = 128) -> None:
        # Imported here, not at module scope: this package's facade imports every encoder so that a
        # name resolves, and `transformers` costs a measured 6.1 s to import — a run that reads no
        # text would pay it for a name it never writes. Same reason the adapter defers `peft`.
        from transformers import AutoTokenizer

        if max_length <= 0:
            raise ValueError(f"max_length is how many tokens a caption becomes, so it is positive, got {max_length}.")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        self.reads: tuple[str, ...] = tuple(self.tokenizer.model_input_names)
        """The names this family's model is called with, which are the leaves of this input's tree."""

    @property
    def info(self) -> InputInfo:
        length = TensorShape(axes=(Axis.TOKENS,), sizes=(self.max_length,))
        return InputInfo(shape={name: length for name in self.reads}, modality=Modality.TEXT)

    def encode(self, value: object) -> Mapping[str, Tensor]:
        """One tensor per name the family reads, each the declared length, for one caption.

        A cell holding no caption is named here rather than handed on: a tokenizer reads an empty
        string as a sentence of nothing but markers, so a column with holes in it would train on rows
        that say nothing at all and report the same numbers as one that is filled.
        """
        if not isinstance(value, str) or not value.strip():
            arrived = repr(value) if isinstance(value, str) else type(value).__name__
            raise ValueError(
                f"A text input has no text to read in {arrived}: every row of this column holds one caption, "
                "and this one holds nothing. Fill the column, or name a filled one in `data.inputs`."
            )
        encoded = self.tokenizer(value, padding="max_length", truncation=True, max_length=self.max_length)
        return {name: torch.as_tensor(encoded[name], dtype=torch.long) for name in self.reads}
