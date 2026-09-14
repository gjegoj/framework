"""Parameters a run adds to a network it did not build, and folds back into it before the run ends."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
from torch import nn

from src.config import ComponentConfig
from src.config.instantiate import resolve_factory
from src.core import Axis, Stream, require_tensor, submodule_at
from src.models import Adapter, CompositeModel, HeadConnection, TimmBackbone
from src.models.build import build_adapter
from src.models.registry import adapter_registry
from tests.unit.models.conftest import POOLED_WIDTH, Encoder

TARGET = "projection"
"""The one part of the test backbone a low-rank delta can be added to: ``backbone.projection``."""

TASK = "label"


def composite() -> CompositeModel:
    """A fresh graph whose weights are the same every time, so two of them are comparable."""
    torch.manual_seed(0)
    connection = HeadConnection(nn.Linear(POOLED_WIDTH, 2), streams=(Stream.POOLED,))
    return CompositeModel(Encoder(), {TASK: connection})


def declaration(**written: Any) -> ComponentConfig:
    return ComponentConfig.model_validate(
        {"name": "lora", "module": "backbone", "target_modules": [TARGET], "r": 2, **written}
    )


def attached(model: CompositeModel, **written: Any) -> Adapter:
    built = build_adapter(declaration(**written), model)
    assert built is not None
    return built


def delta_of(model: nn.Module, plain: nn.Module) -> list[nn.Parameter]:
    """Everything the adapted network holds that an unadapted one does not, which is the delta itself."""
    parameters = dict(model.named_parameters())
    added = sorted(set(parameters) - set(dict(plain.named_parameters())))
    assert added, "nothing was added, so there is no delta for this test to be about"
    return [parameters[name] for name in added]


def learn(model: nn.Module, plain: nn.Module) -> None:
    """Move the delta off the zero it is born at, the way a first epoch would."""
    torch.manual_seed(1)
    with torch.no_grad():
        for parameter in delta_of(model, plain):
            parameter.normal_(0.0, 0.5)


def answer(model: CompositeModel, images: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return require_tensor(model(images).outputs[TASK], name=TASK)


def test_the_delta_learns_and_the_weights_it_was_added_to_stop_learning() -> None:
    """Adapting is holding a network still and learning a small change beside it; the heads are not part of it."""
    model, plain = composite(), composite()
    attached(model)

    learning = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    held = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    ordinary = set(dict(plain.named_parameters()))
    heads = {name for name in ordinary if name.startswith("heads.")}
    # Counted rather than named: adapting renames the weights it goes beside, so a weight left learning
    # is not recognisable by having a name the unadapted network also has.
    added = len(dict(model.named_parameters())) - len(ordinary)

    assert heads <= learning, "the heads are not what is being adapted and go on learning"
    assert len(learning - heads) == added > 0, "something other than what was added is still learning"
    assert all(name.startswith("backbone.") for name in held), "something outside the adapted module was held still"


def test_every_path_a_declaration_writes_still_names_what_it_named() -> None:
    """``backbone`` and ``heads.<task>`` are what a freeze callback, a checkpoint and a parameter group address."""
    model = composite()
    encoder = submodule_at(model, "backbone", reader="test")
    attached(model)

    assert submodule_at(model, "backbone", reader="test") is encoder
    assert submodule_at(model, f"heads.{TASK}", reader="test") is model.heads[TASK]
    assert submodule_at(model, f"backbone.{TARGET}", reader="test") is not None


def test_a_network_answers_the_same_the_moment_a_delta_is_attached(images: dict[str, torch.Tensor]) -> None:
    """A run that adapts pretrained weights starts from what they already answered, not beside it."""
    model, plain = composite(), composite()
    attached(model)

    assert torch.allclose(answer(model, images), answer(plain, images))


def test_merging_keeps_the_answer_the_delta_had_learned(images: dict[str, torch.Tensor]) -> None:
    """What is evaluated and shipped after the fold is the model the run stopped on, not the one it started from."""
    model, plain = composite(), composite()
    adapter = attached(model)
    learn(model, plain)
    learned = answer(model, images)
    assert not torch.allclose(learned, answer(plain, images)), "the delta changed nothing, so folding it proves nothing"

    adapter.merge()

    assert torch.allclose(learned, answer(model, images), atol=1e-6)


def test_after_merging_the_network_is_named_as_one_that_was_never_adapted() -> None:
    """A checkpoint and an artifact of an adapted run are readable by a run that declares no adapters."""
    model, plain = composite(), composite()
    adapter = attached(model)
    assert list(model.state_dict()) != list(plain.state_dict())

    adapter.merge()

    assert list(model.state_dict()) == list(plain.state_dict())


def test_a_target_that_matches_nothing_under_the_module_names_the_parts_that_are_there() -> None:
    """``target_modules`` has no safe default, and a name matching nothing would adapt a network by zero parts."""
    with pytest.raises(ValueError) as refusal:
        attached(composite(), target_modules=["qkv"])

    assert "adapter.target_modules" in str(refusal.value)
    assert TARGET in str(refusal.value)


def test_the_parts_a_refusal_offers_are_the_ones_a_delta_could_go_beside() -> None:
    """A normalisation's weight is a vector and takes no low-rank delta; offering it would send a reader to try it.

    Read on a real backbone, because the one this module builds has a single part and could not tell a
    rule about weight matrices from a list of every name in the tree.
    """
    torch.manual_seed(0)
    encoder = TimmBackbone(model_name="resnet18", pretrained=False)
    width = encoder.feature_shapes[Stream.POOLED].size(Axis.CHANNELS) or 0
    model = CompositeModel(encoder, {TASK: HeadConnection(nn.Linear(width, 2), streams=(Stream.POOLED,))})

    with pytest.raises(ValueError) as refusal:
        build_adapter(declaration(target_modules=["qkv"]), model)

    assert "conv1, conv2" in str(refusal.value)
    assert "bn1" not in str(refusal.value)


def test_a_module_this_model_does_not_have_is_refused_with_the_ones_it_does() -> None:
    with pytest.raises(LookupError) as refusal:
        attached(composite(), module="encoder")

    assert "backbone, heads" in str(refusal.value)


def test_a_run_that_declares_no_adapters_adds_none() -> None:
    """``adapters: none`` is a declaration, and the run says so by having nothing to fold back."""
    assert build_adapter(None, composite()) is None


def test_something_that_is_not_an_adapter_is_refused_where_it_was_declared() -> None:
    with pytest.raises(TypeError, match="Counter"):
        build_adapter(ComponentConfig.model_validate({"_target_": "collections.Counter"}), composite())


def test_the_name_resolves_long_before_the_library_behind_it_is_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing this package registers every adapter; a run that declares none pays for none."""
    monkeypatch.setitem(sys.modules, "peft", None)

    resolved = resolve_factory(ComponentConfig(name="lora"), adapter_registry)

    assert isinstance(resolved, type) and issubclass(resolved, Adapter)
    with pytest.raises(ImportError):
        build_adapter(declaration(), composite())
