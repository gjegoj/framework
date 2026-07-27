"""LoraConfig validation: targets required, positive rank/alpha, extras forwarded."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.schema import LoraConfig


class TestLoraConfig:
    def test_minimal_valid(self) -> None:
        config = LoraConfig(target_modules=["qkv"])
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.0

    def test_empty_targets_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LoraConfig(target_modules=[])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("rank", 0, id="zero_rank"),
            pytest.param("alpha", 0.0, id="zero_alpha"),
            pytest.param("dropout", 1.0, id="dropout_at_one"),
            pytest.param("dropout", -0.1, id="negative_dropout"),
        ],
    )
    def test_invalid_numbers_rejected(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            LoraConfig(**{"target_modules": ["qkv"], field: value})

    def test_extras_kept_for_forwarding(self) -> None:
        """Unknown keys forward verbatim to peft's LoraConfig (use_dora, use_rslora, bias, ...)."""
        config = LoraConfig(target_modules=["qkv"], use_dora=True)
        assert config.model_extra == {"use_dora": True}

    def test_experiment_config_defaults_to_disabled(self) -> None:
        from src.config.schema import ExperimentConfig

        assert ExperimentConfig.model_fields["lora"].default is None
