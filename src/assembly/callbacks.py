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

    One grammar builds them. ``tasks`` and ``profile`` are offered to every entry and reach
    only the ones that name them. Only facts assembly alone knows travel this way: a shared
    config value (``mean``, ``lr``) is reached by interpolation, ``${mean}``, as the
    transforms reach it — offered here it would silently override a per-callback value.
    """
    if config.callbacks is None:
        return []
    derived = {"tasks": list(tasks), "profile": profile if profile is not None else DataProfile()}
    built: list[Callback] = [instantiate(declared, callback_registry, **derived) for declared in config.callbacks]
    return built
