"""Shared wiring support: the ``WiringContext`` value object and an extras helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.config.schema import ExperimentConfig
from src.core.runtime import RuntimeContext


@dataclass(frozen=True)
class WiringContext:
    """The validated config plus populated runtime, for context-aware builders.

    A convenient form of the data crossing into component builders that need
    cross-references (a task's ``num_classes`` from ``runtime``, ``save_dir`` from
    ``config``): one typed value instead of threading two positional args.

    Parameters:
        config (ExperimentConfig): The validated experiment config.
        runtime (RuntimeContext): Context populated by ``DataModule.setup()``.
    """

    config: ExperimentConfig
    runtime: RuntimeContext


def forward_extras(section: BaseModel, core_fields: frozenset[str]) -> dict[str, Any]:
    """Return a typed config section's non-core fields, to forward as kwargs.

    The ``core_fields`` are passed explicitly by the section's builder; every
    other field (adapter-specific extras such as smp's ``encoder_name`` or an
    optimizer's ``momentum``) is forwarded verbatim.

    Parameters:
        section (BaseModel): A validated config section (extras allowed).
        core_fields (frozenset[str]): Field names handled explicitly by the builder.

    Returns:
        dict[str, Any]: The remaining fields, keyed by name.
    """
    return {key: value for key, value in section.model_dump().items() if key not in core_fields}
