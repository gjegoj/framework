"""Building the callbacks of a run from config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.assembly.instantiate import instantiate
from src.callbacks.registry import callback_registry
from src.core.entities import DataProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning.pytorch.callbacks import Callback

    from src.config import ExperimentConfig
    from src.core.entities import Task


def build_callbacks(
    config: ExperimentConfig,
    tasks: Sequence[Task] = (),
    profile: DataProfile | None = None,
) -> list[Callback]:
    """The declared callbacks, in the order the file gives them.

    One grammar builds them, as it builds every other component. ``tasks`` and
    ``profile`` are offered to every entry and reach only the ones that name
    them — a batch transform has to rewrite each task's label, while a
    checkpoint has no use for either.

    Only facts assembly alone knows travel this way. A shared *config* value —
    ``mean``, ``lr``, ``image_size`` — has its own way of reaching a component
    and it is ``${mean}``: the same interpolation ``configs/transforms/*.yaml``
    already uses. Offering one here was tried and reverted; a derived value
    overrides what config declares, so a per-callback ``mean:`` was accepted and
    then silently ignored. That is the whole of what a
    "context-aware callback" needs: no second registry of builders, no special
    case here.
    """
    if config.callbacks is None:
        return []
    derived = {"tasks": list(tasks), "profile": profile if profile is not None else DataProfile()}
    built: list[Callback] = [instantiate(declared, callback_registry, **derived) for declared in config.callbacks]
    return built
