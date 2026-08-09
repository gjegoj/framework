"""The adapters section, turned into a transformation of the model that is built."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.assembly.models import build_model
from src.config import ExperimentConfig
from src.models import CompositeModel
from tests.support.configs import disk_config
from tests.support.entities import profiled

LORA = {"name": "lora", "target_modules": ["fc1", "fc2"], "rank": 4}

VIT = {"name": "timm", "model_name": "vit_tiny_patch16_224", "pretrained": False, "img_size": 16}
"""A ViT carries the module names LoRA targets; 16 matches the shared transforms."""


def experiment(root: Path, **overrides: Any) -> ExperimentConfig:
    """The shared on-disk experiment on a ViT, whose module names LoRA can target."""
    return disk_config(root, model=VIT, **overrides)


def test_no_adapters_section_leaves_every_weight_learning(dataset_root: Path) -> None:
    """The default has to stay full fine-tuning; adapters are opted into."""
    model, _ = build_model(experiment(dataset_root), profiled())

    assert isinstance(model, CompositeModel)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_the_declared_technique_is_applied_to_the_backbone(dataset_root: Path) -> None:
    """Config names a technique; what arrives is a backbone whose base is already held still."""
    model, _ = build_model(experiment(dataset_root, adapters=LORA), profiled())

    assert isinstance(model, CompositeModel)
    trainable = {name for name, parameter in model.backbone.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_the_heads_still_learn(dataset_root: Path) -> None:
    """LoRA freezes the backbone, not the task: a head that cannot move learns nothing at all."""
    model, _ = build_model(experiment(dataset_root, adapters=LORA), profiled())

    assert isinstance(model, CompositeModel)
    assert any(parameter.requires_grad for parameter in model.heads.parameters())


def test_a_freeze_callback_aimed_at_the_backbone_is_refused(dataset_root: Path) -> None:
    """It would hold the adapters still too, and training would run with nothing to learn."""
    callbacks = [{"name": "freeze", "modules": ["model.backbone"]}]

    with pytest.raises(ValueError, match="freeze"):
        build_model(experiment(dataset_root, adapters=LORA, callbacks=callbacks), profiled())


def test_a_freeze_callback_elsewhere_is_left_alone(dataset_root: Path) -> None:
    """Only the backbone is contested; holding another part still is a separate, legal decision."""
    callbacks = [{"name": "freeze", "modules": ["model.heads"]}]

    model, _ = build_model(experiment(dataset_root, adapters=LORA, callbacks=callbacks), profiled())

    assert model is not None
