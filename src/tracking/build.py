"""A `tracker` declaration becomes the place a run is recorded."""

from __future__ import annotations

from lightning.pytorch.loggers import Logger

from src.config import ComponentConfig
from src.config.instantiate import instantiate
from src.tracking.registry import tracker_registry


def build_tracker(declared: ComponentConfig | None) -> Logger | None:
    """Where this run's numbers go, or nothing at all — `tracker: none` is a declaration, not a gap."""
    if declared is None:
        return None
    built = instantiate(declared, tracker_registry)
    if not isinstance(built, Logger):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which a trainer cannot record to: a "
            "tracker is a Lightning logger, and what it can draw besides numbers it says by its methods."
        )
    return built
