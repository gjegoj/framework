"""A caption becomes one pooled vector: the family reads its own tree, and padding never gets a vote."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.core import Axis, Stream, TensorShape
from src.data.encoders.text import TextEncoder
from src.models.backbones.hf import HFTextBackbone
from src.models.registry import backbone_registry
from tests.support.text import WIDTH, text_family

LENGTH = 8
CAPTION = "a brown dog on the porch"


@pytest.fixture
def family(tmp_path: Path) -> Path:
    return text_family(tmp_path / "family")


def tree(family: Path, caption: str, length: int = LENGTH) -> dict[str, Tensor]:
    """One caption as the batch of one a run's collator makes of it, encoded by the same family."""
    encoded = TextEncoder(str(family), max_length=length).encode(caption)
    return {name: value.unsqueeze(0) for name, value in encoded.items()}


def test_it_publishes_one_pooled_vector_sized_by_the_family(family: Path) -> None:
    """A head over text is sized like a head over pixels: one width, from the backbone's own declaration."""
    backbone = HFTextBackbone(str(family))

    encoded = backbone({"text": tree(family, CAPTION)})

    assert backbone.feature_shapes == {Stream.POOLED: TensorShape(axes=(Axis.CHANNELS,), sizes=(WIDTH,))}
    assert tuple(encoded[Stream.POOLED].shape) == (1, WIDTH)


def test_a_caption_reads_the_same_however_much_padding_follows_it(family: Path) -> None:
    """Padding is a stacking convenience, so it may not move the answer; the mask is what says so.

    Measured on a family this size: an unmasked mean of the same sentence at two lengths differs by more
    than a unit, and keeps drifting as the declared length grows — the same caption would mean something
    else in a run that declared room for longer ones.
    """
    backbone = HFTextBackbone(str(family), pooling="mean").eval()

    with torch.no_grad():
        short = backbone({"text": tree(family, CAPTION, length=8)})[Stream.POOLED]
        padded = backbone({"text": tree(family, CAPTION, length=24)})[Stream.POOLED]

    assert torch.allclose(short, padded, atol=1e-6)


def test_the_marker_pooling_reads_the_position_the_family_marks(family: Path) -> None:
    """``cls`` is the vector the family puts first, which is what its own pretraining summarises with."""
    backbone = HFTextBackbone(str(family), pooling="cls").eval()
    tokens = tree(family, CAPTION)

    with torch.no_grad():
        pooled = backbone({"text": tokens})[Stream.POOLED]
        first = backbone.model(**tokens).last_hidden_state[:, 0]

    assert torch.equal(pooled, first)


def test_an_input_that_is_not_the_tree_this_family_reads_is_refused_by_name(family: Path) -> None:
    """A backbone pointed at the pictures instead of the captions says which input it was given."""
    with pytest.raises(TypeError, match="named tensors"):
        HFTextBackbone(str(family))({"text": torch.zeros(1, 3, 8, 8)})


def test_a_caption_with_nothing_saying_which_tokens_are_words_is_refused_rather_than_averaged(family: Path) -> None:
    """A mean with no mask counts padding, and the number it makes is indistinguishable from a real one.

    Matched on what only this refusal says: a bare miss on the key raises ``KeyError``, which *is* a
    ``LookupError`` spelling the name back — so the first version of this test passed with the refusal
    deleted.
    """
    ids = tree(family, CAPTION)["input_ids"]

    with pytest.raises(LookupError, match="which positions are words"):
        HFTextBackbone(str(family), pooling="mean")({"text": {"input_ids": ids}})


def test_a_pooling_nobody_implements_is_refused_with_the_ones_there_are(family: Path) -> None:
    with pytest.raises(ValueError, match="cls, mean"):
        HFTextBackbone(str(family), pooling="max")


def test_the_name_resolves_long_before_the_library_behind_it_is_needed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Importing this package registers every backbone; a run over pixels pays for none of transformers."""
    monkeypatch.setitem(sys.modules, "transformers", None)

    assert issubclass(backbone_registry.get("hf_text"), HFTextBackbone)
    with pytest.raises(ImportError):
        HFTextBackbone(str(tmp_path))
