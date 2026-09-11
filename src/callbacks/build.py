"""The `callbacks` section as the list a trainer is handed."""

from __future__ import annotations

from collections.abc import Sequence

from lightning.pytorch.callbacks import Callback

from src.callbacks.registry import callback_registry
from src.config import ComponentConfig
from src.config.instantiate import instantiate


def build_callbacks(declared: Sequence[ComponentConfig]) -> list[Callback]:
    """Every declared callback, in the order the file gives them — order is a decision a run makes.

    Built from their declarations alone: one that needs the run's tasks reads them off the module in
    ``setup``, the way any Lightning callback does, so nothing is offered here by name.
    """
    built = [instantiate(one, callback_registry) for one in declared]
    strangers = [
        f"{one.spelled} built {type(made).__name__}"
        for one, made in zip(declared, built, strict=True)
        if not isinstance(made, Callback)
    ]
    if strangers:
        raise TypeError(f"A callback hooks into the run's loop, and {', '.join(strangers)}, which has no hooks to run.")
    return built
