"""``LoraAdapters``: a low-rank delta learns while the weights it stands in for hold still."""

from __future__ import annotations

import pytest
import timm
import torch
from torch import nn

from src.models import LoraAdapters, merge_adapters
from src.models.registry import adapter_registry

TINY = "vit_tiny_patch16_224"


def backbone() -> nn.Module:
    """A ViT small enough to build in a test, with the module names LoRA targets."""
    return timm.create_model(TINY, pretrained=False, num_classes=0)


def test_only_the_named_modules_gain_adapters() -> None:
    """`target_modules` is the one thing that cannot be guessed, so it must be obeyed exactly."""
    model = backbone()

    LoraAdapters(target_modules=["qkv"])(model)

    adapted = {name.rsplit(".lora_A", 1)[0] for name, _ in model.named_parameters() if ".lora_A" in name}
    assert adapted
    assert all(name.endswith("qkv") for name in adapted)


def test_everything_but_the_adapters_stops_learning() -> None:
    """The base holding still is the technique; a base that keeps training is just slow fine-tuning."""
    model = backbone()

    LoraAdapters(target_modules=["qkv", "proj"])(model)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("lora_" in name for name in trainable)


def test_a_target_that_matched_nothing_is_refused_by_name() -> None:
    """Unrefused, a misspelt target trains the whole backbone at full cost and looks like it worked."""
    with pytest.raises(ValueError, match="attention"):
        LoraAdapters(target_modules=["attention"])(backbone())


def test_the_fold_leaves_the_weights_named_as_if_peft_never_ran() -> None:
    """Measured: injection renames 50 of this model's keys; the fold is what makes a run's output plain."""
    plain = set(backbone().state_dict())
    model = backbone()
    LoraAdapters(target_modules=["qkv", "proj"])(model)
    assert set(model.state_dict()) != plain

    merge_adapters(model)

    assert set(model.state_dict()) == plain


def test_the_fold_does_not_change_what_the_model_computes() -> None:
    """It is exact, which is why `test` and the artifact can share it."""
    model = backbone().eval()
    LoraAdapters(target_modules=["qkv"], rank=4)(model)
    example = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        adapted = model(example)

    merge_adapters(model)

    with torch.no_grad():
        assert torch.allclose(model(example), adapted, atol=1e-5)


def test_folding_a_model_without_adapters_does_nothing() -> None:
    """`run()` calls it unconditionally, so it has to be silent on a run that never adapted anything."""
    model = backbone()
    before = model.state_dict()

    assert merge_adapters(model) == 0
    assert set(model.state_dict()) == set(before)


def test_the_technique_is_reachable_from_config_by_name() -> None:
    """Config names a plug-in; the registry is how the name becomes an object."""
    built = adapter_registry.create("lora", target_modules=["qkv"], rank=4)

    assert isinstance(built, LoraAdapters)
    assert built.rank == 4
