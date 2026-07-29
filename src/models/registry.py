"""Registries for pluggable model components (extension points for users)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import Backbone, Head
    from src.models.complete import CompleteModel

backbones: Registry[Backbone] = Registry("backbone")
head_builders: Registry[Head] = Registry("head")
complete_models: Registry[CompleteModel[Any, Any]] = Registry("complete model")


def register_complete_model(
    kind: str,
) -> Callable[[Callable[..., CompleteModel[Any, Any]]], Callable[..., CompleteModel[Any, Any]]]:
    """Register a complete-model family under ``kind``, keeping ``kind`` one namespace.

    The model config section's ``kind`` selects either an assembled backbone or a
    complete model; a key living in both registries would make dispatch ambiguous,
    so collisions fail at import time. (Backbones register first in import order,
    which is why the check runs on this side.)

    Parameters:
        kind (str): Model-section ``kind`` key, e.g. ``"yolo"``.

    Raises:
        ValueError: If ``kind`` is already a backbone kind.
    """
    if kind in backbones:
        raise ValueError(
            f"complete model kind {kind!r} is already a backbone kind — "
            "the model section's 'kind' is a single (kind) namespace."
        )
    return complete_models.register(kind)
