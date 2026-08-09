"""``ComponentConfig``: the one grammar for configuring a named component."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import ComponentConfig


def test_a_bare_string_is_a_registry_name() -> None:
    component = ComponentConfig.model_validate("cross_entropy")

    assert component.name == "cross_entropy"
    assert component.params == {}


def test_extra_keys_become_constructor_params() -> None:
    component = ComponentConfig.model_validate({"name": "cross_entropy", "label_smoothing": 0.1})

    assert component.name == "cross_entropy"
    assert component.params == {"label_smoothing": 0.1}


def test_target_is_the_escape_hatch_for_unregistered_code() -> None:
    component = ComponentConfig.model_validate({"_target_": "my_pkg.FocalLoss", "gamma": 2.0})

    assert component.target == "my_pkg.FocalLoss"
    assert component.params == {"gamma": 2.0}


@pytest.mark.parametrize(
    "raw",
    [{"name": "ce", "_target_": "my_pkg.Loss"}, {"label_smoothing": 0.1}],
    ids=["both", "neither"],
)
def test_a_component_names_exactly_one_thing_to_build(raw: dict[str, object]) -> None:
    """Two references contradict each other and none names nothing: neither is buildable."""
    with pytest.raises(ValidationError, match="exactly one"):
        ComponentConfig.model_validate(raw)


def test_hydra_meta_keys_are_rejected_instead_of_ignored() -> None:
    """Dropping these silently is how a user asks for a factory and gets an instance."""
    with pytest.raises(ValidationError, match="_partial_"):
        ComponentConfig.model_validate({"_target_": "torch.optim.AdamW", "_partial_": True})


def test_the_rejection_message_names_every_reserved_key() -> None:
    with pytest.raises(ValidationError, match="_convert_, _recursive_"):
        ComponentConfig.model_validate({"_target_": "torch.optim.AdamW", "_recursive_": False, "_convert_": "all"})
