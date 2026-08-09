"""``HFTextBackbone``: a transformers text encoder as a pooled-embedding backbone."""

from __future__ import annotations

import pytest
import torch

from src.core import Stream
from src.models import HFTextBackbone
from src.models.registry import backbone_registry

TINY_BERT = "hf-internal-testing/tiny-random-bert"


def test_registered_under_the_hf_text_key() -> None:
    assert "hf_text" in backbone_registry


def test_a_misspelt_pooling_is_refused_before_a_model_is_fetched() -> None:
    """Unchecked, 'men' fell through to the mean branch — a silently different model.

    No network: the refusal comes before the weights, which is also why a typo
    costs a message rather than a download.
    """
    with pytest.raises(ValueError, match="Pooling must be one of cls, mean"):
        HFTextBackbone(model_name="never-fetched", pooling="men")  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def backbone() -> HFTextBackbone:
    return HFTextBackbone(model_name=TINY_BERT, pretrained=False)


@pytest.mark.slow
def test_encodes_token_ids_into_a_pooled_embedding(backbone: HFTextBackbone) -> None:
    tokens = torch.randint(1, 30, (2, 12))

    features = backbone({"text": tokens})

    assert features[Stream.FEATURES].shape == (2, backbone.feature_dim(Stream.FEATURES))


@pytest.mark.slow
def test_attention_mask_input_changes_mean_pooling(backbone: HFTextBackbone) -> None:
    pooled = HFTextBackbone(model_name=TINY_BERT, pretrained=False, pooling="mean")
    tokens = torch.randint(1, 30, (1, 6))
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

    with_mask = pooled({"text": tokens, "text_mask": mask})[Stream.FEATURES]
    without_mask = pooled({"text": tokens})[Stream.FEATURES]

    assert not torch.allclose(with_mask, without_mask)


@pytest.mark.slow
def test_unknown_stream_is_rejected_by_name(backbone: HFTextBackbone) -> None:
    with pytest.raises(LookupError, match=Stream.FEATURES):
        backbone.feature_dim(Stream.DECODER)
