"""One constructor boundary for short names and import paths, with no registry scanning."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from src.config.schema import ComponentConfig


def resolve_constructor(config: ComponentConfig, names: Mapping[str, str], *, path: str) -> Callable[..., Any]:
    """Resolve only the selected implementation, using the public Hydra resolver lazily."""
    from hydra.utils import get_object

    target = config.import_path
    if target is None:
        if config.name not in names:
            raise ValueError(f"{path}: unknown component {config.name!r}.")
        target = names[cast(str, config.name)]
    try:
        constructor = get_object(target)
    except (ImportError, AttributeError, ValueError) as error:
        raise ValueError(f"{path}: cannot resolve {target!r}: {error}") from error
    if not callable(constructor):
        raise TypeError(f"{path}: {target!r} is not a constructor.")
    return cast(Callable[..., Any], constructor)
