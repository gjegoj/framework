"""A task's declaration becomes its objective: the losses it names, or the one its semantics implies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from typing import Any, cast

from torch import nn

from src.config import ComponentConfig, WeightedLossConfig
from src.config.instantiate import instantiate_offering
from src.losses.base import Loss, NamedLoss
from src.losses.composite import WeightedSum
from src.losses.registry import loss_registry


def build_loss(declared: object, facts: Mapping[str, Any]) -> Loss:
    """One task's objective, from the declaration that settled it and what the run settled about its target.

    Which declaration that is — the run's or the task's own default — is decided before this call, so
    this package needs to know nothing about tasks. Facts are never declared: each reaches a loss that
    names it in its constructor, and a declaration restating one is refused by name.
    """
    parts = [(_one(part.loss, facts, part.log_name), part.weight) for part in _weighted(declared)]
    if len(parts) == 1 and parts[0][1] == 1.0:
        return parts[0][0]
    return WeightedSum(parts)


def _weighted(declared: object) -> list[WeightedLossConfig]:
    """One grammar out of the several a declaration may use: a name, one component, or a weighted list.

    A task's own default is written the same way a run writes one, so both arrive here and leave alike.
    """
    if isinstance(declared, ComponentConfig):
        return [WeightedLossConfig(loss=declared)]
    if isinstance(declared, str | Mapping):
        return [WeightedLossConfig.model_validate({"loss": declared})]
    if isinstance(declared, Sequence):
        return [
            one if isinstance(one, WeightedLossConfig) else WeightedLossConfig.model_validate(one) for one in declared
        ]
    raise TypeError(f"A loss is declared as a name, a component or a weighted list; got {type(declared).__name__}.")


def _one(declared: ComponentConfig, facts: Mapping[str, Any], log_name: str | None) -> Loss:
    built: Any = instantiate_offering(declared, loss_registry, **facts)
    if not isinstance(built, Loss):
        if not (isinstance(built, nn.Module) and _compares_two_tensors(built)):
            raise TypeError(
                f"{declared.spelled!r} built {type(built).__name__}, which does not compare an output with a "
                "target: a loss takes both."
            )
        built = NamedLoss(built)
    # A term reports under the name the run wrote for it, which is what a metric key then shows.
    built.log_name = log_name or declared.name or built.log_name
    return cast(Loss, built)


def _compares_two_tensors(module: nn.Module) -> bool:
    """A loss's forward takes the output and the target; anything taking one is some other kind of module."""
    accepted = signature(type(module).forward).parameters
    positional = [name for name, one in accepted.items() if one.kind is not one.KEYWORD_ONLY][1:]  # drop self
    return len(positional) >= 2
