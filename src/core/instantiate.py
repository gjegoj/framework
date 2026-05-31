"""Generic spec resolver: turn a YAML config fragment into a concrete object.

This is the single mechanism that restores the prototype's YAML mobility while
keeping the validated-boundary discipline. A *brick spec* selects a component in
one of three forms, in increasing order of power:

- **string** — a registry key with default args: ``cross_entropy``.
- **mapping with** ``name`` — a registry key plus keyword arguments:
  ``{name: cross_entropy, label_smoothing: 0.1}``.
- **mapping with** ``_target_`` — a fully-qualified import path for a custom
  class the framework has never heard of: ``{_target_: my_pkg.MyLoss, alpha: 0.3}``.

``injected`` arguments are runtime-derived values the caller forces (e.g. a
metric's ``num_classes``); user-supplied params take precedence so a config can
still override them when it makes sense.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from src.core.registry import Registry

BrickSpec = str | Mapping[str, Any]


def resolve_target(path: str) -> Any:
    """Import and return the object at a dotted ``module.attr`` path.

    Parameters:
        path (str): Fully-qualified path, e.g. ``"torch.optim.SGD"``.

    Returns:
        Any: The imported class/callable.

    Raises:
        ValueError: If ``path`` has no module component.
    """
    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ValueError(f"_target_ must be a dotted path 'module.attr', got {path!r}.")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def instantiate[T](spec: BrickSpec, registry: Registry[T], **injected: Any) -> T:
    """Build a component from a brick spec via ``registry`` or a ``_target_`` import.

    Parameters:
        spec (BrickSpec): String key, or mapping with ``name``/``_target_`` + kwargs.
        registry (Registry[T]): Registry consulted for the string/``name`` forms.
        **injected (Any): Caller-forced defaults (overridable by spec params).

    Returns:
        T: The constructed component.

    Raises:
        ValueError: If a mapping spec has neither ``name`` nor ``_target_``.
        TypeError: If ``spec`` is neither a string nor a mapping.
    """
    if isinstance(spec, str):
        return registry.create(spec, **injected)
    if isinstance(spec, Mapping):
        params = dict(spec)
        target = params.pop("_target_", None)
        if target is not None:
            factory = resolve_target(str(target))
            built: T = factory(**{**injected, **params})
            return built
        name = params.pop("name", None)
        if name is None:
            raise ValueError(f"Brick spec mapping needs a 'name' or '_target_' key: {spec!r}.")
        return registry.create(str(name), **{**injected, **params})
    raise TypeError(f"Brick spec must be a string or mapping, got {type(spec).__name__}.")
