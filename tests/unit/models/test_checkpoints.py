"""Weights go into a model exactly — beneath its adapters when it wears some, refused otherwise."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.adapters import LoraAdapters
from src.models.checkpoints import load_weights


class Tiny(nn.Module):
    """One linear layer — the smallest model whose weights can be told apart."""

    def __init__(self, fill: float) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)
        with torch.no_grad():
            self.net.weight.fill_(fill)


def adapted(fill: float) -> Tiny:
    model = Tiny(fill)
    LoraAdapters(target_modules=["net"], rank=2)(model)
    return model


def test_plain_weights_load_beneath_freshly_added_adapters() -> None:
    """Warm-starting LoRA from a plain run's weights is the ordinary way to reach for it.

    The base loads under the adapters and the deltas keep their zero start, so the
    adapted model computes exactly what the weights say until training moves the delta."""
    weights = Tiny(1.0).state_dict()
    model = adapted(0.0)

    load_weights(model, weights, "plain.ckpt")

    grafted = model.state_dict()["net.base_layer.weight"]
    assert torch.allclose(grafted, torch.full_like(grafted, 1.0))
    x = torch.randn(5, 4)
    assert torch.allclose(model.net(x), torch.nn.functional.linear(x, weights["net.weight"], weights["net.bias"]))


def test_an_adapted_runs_weights_still_load_strictly_into_an_adapted_model() -> None:
    """The evaluate-a-lora-checkpoint workflow: adapted keys fit an adapted model as-is."""
    model = adapted(0.0)

    load_weights(model, adapted(1.0).state_dict(), "lora.ckpt")

    grafted = model.state_dict()["net.base_layer.weight"]
    assert torch.allclose(grafted, torch.full_like(grafted, 1.0))


def test_weights_of_a_different_architecture_are_refused_even_beneath_adapters() -> None:
    """The graft fixes exactly one mismatch — the adapters' rename — and nothing else."""
    with pytest.raises(ValueError, match="beneath the adapters"):
        load_weights(adapted(0.0), nn.Linear(8, 8).state_dict(), "foreign.ckpt")


def test_a_size_mismatch_is_refused_by_name_even_beneath_adapters() -> None:
    """A layer that grew a different width dies with the framework's own refusal, not a raw
    size-mismatch traceback: the graft renames keys, and a shape that disagrees after the
    rename is the architecture disagreeing."""

    class Wide(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Linear(4, 20)

    with pytest.raises(ValueError, match="beneath the adapters"):
        load_weights(adapted(0.0), Wide().state_dict(), "wide.ckpt")


def test_weights_that_do_not_fit_name_the_model_and_the_source() -> None:
    """Two shapes are refused here — an adapted run's and a mismatched teacher's — so the message names both."""
    with pytest.raises(ValueError, match=r"plain.ckpt does not fit Linear"):
        load_weights(nn.Linear(8, 8), Tiny(1.0).state_dict(), "plain.ckpt")


def test_a_checkpoint_of_a_wrapped_native_head_is_refused_naming_the_two_spellings() -> None:
    """Native heads once sat under ``heads.<task>._module``; a checkpoint of that spelling is
    refused like any other misfit, and the message says where those keys live now."""
    with pytest.raises(ValueError, match=r"heads\.<task>\._module"):
        load_weights(Tiny(0.0), {"heads.t._module.weight": torch.zeros(2, 4)}, source="old.ckpt")
