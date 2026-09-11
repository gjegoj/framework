"""A task's declaration becomes its objective: the losses it names, or the one its semantics implies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from typing import Any, cast

from torch import Tensor, nn

from src.config import ComponentConfig, WeightedLossConfig
from src.config.instantiate import fill_signature, resolve_params, resolve_target
from src.core import LossOutput, TargetInfo
from src.losses.base import Loss, snake_case
from src.losses.composite import WeightedSum
from src.losses.registry import loss_registry

type Declared = ComponentConfig | Sequence[WeightedLossConfig] | None
"""What a run writes under `tasks.<name>.loss`: one component, several with weights, or nothing at all."""


def build_loss(declared: object, info: TargetInfo) -> Loss:
    """One task's objective, from the declaration that settled it and the facts its target encoder left.

    Which declaration that is — the run's or the task's own default — is decided before this call, so
    this package needs to know nothing about tasks. Sizes are never declared: facts the encoder settled
    reach a loss that names them in its constructor, and a declaration restating one is refused by name.
    """
    parts = [(_one(part.loss, info, part.log_name), part.weight) for part in _weighted(declared)]
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


def _one(declared: ComponentConfig, info: TargetInfo, log_name: str | None) -> Loss:
    factory = resolve_target(declared, loss_registry)
    facts: dict[str, Any] = {"values": info.values, "num_classes": info.num_classes}
    restated = sorted(fill_signature(factory, **facts).keys() & declared.params.keys())
    if restated:
        raise ValueError(
            f"{declared.spelled!r} declares {', '.join(restated)}, which the target encoder already settled; "
            "drop it from the declaration."
        )
    built: Any = factory(**resolve_params(declared), **fill_signature(factory, **facts))
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


class NamedLoss(Loss):
    """A torch loss reached by import path: it reports under its own class name unless a run renames it."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.log_name = snake_case(type(module).__name__.removesuffix("Loss")) or "loss"

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        return self.reported(cast(Tensor, self.module(outputs, targets)))
