"""Trainable-only checkpoint pruning: frozen params dropped, buffers and trainables kept."""

from __future__ import annotations

import torch

from src.models import build_composite_model
from src.models.backbones import EmbeddingBackbone
from src.tasks import classification
from src.training import LitModule
from src.training.optim import OptimizerBuilder


def _module_with_frozen_parameter(checkpoint_trainable_only: bool) -> tuple[LitModule, str, str]:
    """Build a module, freeze one head parameter; return (module, frozen_key, trainable_key)."""
    task = classification("label", num_classes=3)
    model = build_composite_model(EmbeddingBackbone(embedding_dim=8), {"label": task.head_spec})
    module = LitModule(
        model=model,
        tasks=[task],
        optimizer_builder=OptimizerBuilder(base_lr=1e-3),
        checkpoint_trainable_only=checkpoint_trainable_only,
    )
    head_parameters = list(module.model.heads["label"].named_parameters())
    frozen_name, frozen_parameter = head_parameters[0]
    trainable_name = head_parameters[1][0]
    frozen_parameter.requires_grad = False
    prefix = "model.heads.label."
    return module, f"{prefix}{frozen_name}", f"{prefix}{trainable_name}"


class TestCheckpointPruning:
    def test_disabled_by_default_keeps_everything(self) -> None:
        module, frozen_key, _ = _module_with_frozen_parameter(checkpoint_trainable_only=False)
        checkpoint = {"state_dict": module.state_dict()}
        module.on_save_checkpoint(checkpoint)
        assert frozen_key in checkpoint["state_dict"]
        assert module.strict_loading is True

    def test_prunes_frozen_parameters_only(self) -> None:
        module, frozen_key, trainable_key = _module_with_frozen_parameter(checkpoint_trainable_only=True)
        checkpoint = {"state_dict": module.state_dict(), "current_model_state": module.state_dict()}
        module.on_save_checkpoint(checkpoint)
        for section in ("state_dict", "current_model_state"):
            assert frozen_key not in checkpoint[section], section
            assert trainable_key in checkpoint[section], section
        assert module.strict_loading is False

    def test_roundtrip_with_relaxed_loading(self) -> None:
        module, _, trainable_key = _module_with_frozen_parameter(checkpoint_trainable_only=True)
        checkpoint = {"state_dict": module.state_dict()}
        module.on_save_checkpoint(checkpoint)
        fresh, _, _ = _module_with_frozen_parameter(checkpoint_trainable_only=True)
        fresh.load_state_dict(checkpoint["state_dict"], strict=False)
        restored = dict(fresh.named_parameters())[trainable_key]
        original = dict(module.named_parameters())[trainable_key]
        assert torch.equal(restored, original)
