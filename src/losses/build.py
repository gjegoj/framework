"""A task's declaration becomes its objective: the losses it names, or the one its semantics implies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from typing import Any, cast

from torch import nn

from src.config import ComponentConfig, WeightedLossConfig
from src.config.instantiate import instantiate_offering
from src.core import Representation
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


def refuse_an_objective_the_head_does_not_answer(
    task: str, answered: Representation, loss: Loss, declared_at: str
) -> None:
    """A network and an objective over it are built apart and have to agree about one tensor.

    Here rather than in either builder, because two of them ask it: a task's own objective against the
    head serving it, and the one an algorithm adds against a second network. Why neither the shape nor
    the values tell a projection and an angle apart is ``Representation``, which is the word the two
    declare in — measured on eight real classes, a divergence over unscaled cosines is 256 times the
    one over those same two answers read the way the objective beside them reads them, and a run
    descending the smaller number reports it under a name that reads like work.

    Compared by value rather than by identity: ``Representation`` is a ``StrEnum`` so that a head a run
    wrote itself may spell ``produces = "cosines"`` and be taken at its word.
    """
    if loss.reads == answered:
        return
    raise ValueError(
        f"Task {task!r}: the network serving it answers with {answered}, and objective "
        f"{loss.log_name!r} reads {loss.reads}. Nothing in a tensor says which of the two it holds, so "
        f"this pair would train and report a number that looks like work. Declare `{declared_at}` that "
        f"reads {answered}, or a head that answers with {loss.reads} — `tasks.{task}.head` where the run "
        f"composes one, `produces` on a network arriving whole."
    )


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
