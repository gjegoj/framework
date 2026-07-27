"""LoRA facade: in-place injection, freezing, merge parity, detection."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.config.schema import LoraConfig
from src.models.lora import apply_lora, has_lora_layers, merge_lora


class _TinyBackbone(nn.Module):
    """One linear ('qkv') + one conv ('mix') + one untouched linear ('head_norm')."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.mix = nn.Conv2d(3, 3, kernel_size=1)
        self.head_norm = nn.Linear(8, 8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head_norm(self.qkv(features))  # type: ignore[no-any-return]


def _configured() -> LoraConfig:
    return LoraConfig(target_modules=["qkv", "mix"], rank=2, alpha=4.0)


class TestApplyLora:
    def test_targets_linear_and_conv(self) -> None:
        backbone = _TinyBackbone()
        apply_lora(backbone, _configured())
        assert has_lora_layers(backbone)
        lora_parameter_names = [name for name, _ in backbone.named_parameters() if "lora_" in name]
        assert any(name.startswith("qkv.") for name in lora_parameter_names)
        assert any(name.startswith("mix.") for name in lora_parameter_names)
        assert not any(name.startswith("head_norm.") for name in lora_parameter_names)

    def test_base_frozen_adapters_trainable(self) -> None:
        backbone = _TinyBackbone()
        apply_lora(backbone, _configured())
        for name, parameter in backbone.named_parameters():
            assert parameter.requires_grad == ("lora_" in name), name

    def test_no_match_raises(self) -> None:
        """peft raises its own 'Target modules ... not found'; our post-check is the backstop."""
        with pytest.raises(ValueError, match="(?i)target.modules"):
            apply_lora(_TinyBackbone(), LoraConfig(target_modules=["nonexistent"]))


class TestMergeLora:
    def test_merge_parity_and_clean_tree(self) -> None:
        torch.manual_seed(0)
        backbone = _TinyBackbone()
        apply_lora(backbone, _configured())
        # Give adapters non-zero weights so the merge actually changes W (lora_B init is zeros).
        with torch.no_grad():
            for name, parameter in backbone.named_parameters():
                if "lora_" in name:
                    parameter.add_(torch.randn_like(parameter) * 0.1)
        features = torch.randn(4, 8)
        before = backbone(features)
        merge_lora(backbone)
        after = backbone(features)
        assert not has_lora_layers(backbone)
        assert torch.allclose(before, after, atol=1e-5)
        assert isinstance(backbone.qkv, nn.Linear)  # plain layer, no peft wrapper left
        assert isinstance(backbone.mix, nn.Conv2d)


class TestHasLoraLayers:
    def test_false_on_plain_model(self) -> None:
        assert not has_lora_layers(_TinyBackbone())
