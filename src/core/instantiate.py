"""Generic spec resolver: turn a YAML config fragment into a concrete object.

A *brick spec* selects a component in one of three forms, in increasing order
of power:

- **string** — a registry key with default args: ``cross_entropy``.
- **mapping with** ``name`` — a registry key plus keyword arguments:
  ``{name: cross_entropy, label_smoothing: 0.1}``.
- **mapping with** ``_target_`` — a fully-qualified import path that bypasses
  the registry: ``{_target_: my_pkg.MyLoss, alpha: 0.3}``.

``instantiate`` is also recursive: when a spec has a ``_target_`` key, every
value inside it is resolved the same way before the factory is called.  Lists
are walked element-by-element.  This makes the same function work for both
flat brick selection (losses, metrics, codecs) and deeply nested object graphs
(e.g. an Albumentations ``Compose`` pipeline with ``OneOf`` sub-groups).

Pass a ``registry`` for the string / ``name`` forms; omit it (``None``) for
pure ``_target_``-only specs where no registry exists (third-party libraries).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from src.core.registry import Registry

BrickSpec = str | Mapping[str, Any]

# Hydra meta-keys that carry no meaning outside Hydra — silently dropped.
_HYDRA_META = frozenset({"_convert_", "_recursive_", "_partial_"})


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


def instantiate[T](
    spec: Any,
    registry: Registry[T] | None = None,
    **injected: Any,
) -> T | Any:
    """Build a component from a spec — flat (registry) or recursive (``_target_``).

    Three resolution paths:

    1. **string** — ``registry.create(key, **injected)``; requires ``registry``.
    2. **mapping with** ``name`` — ``registry.create(name, **params)``;
       requires ``registry``.
    3. **mapping with** ``_target_`` — import the class, recursively resolve all
       values in the mapping, then call the class.  Works with or without a
       registry.
    4. **list** — recursively resolve every element.
    5. **scalar** — returned as-is.

    Hydra meta-keys (``_convert_``, ``_recursive_``, ``_partial_``) are dropped
    silently so configs written for Hydra's ``instantiate`` work unchanged.

    Parameters:
        spec (Any): A brick spec (str, mapping, list) or a plain scalar.
        registry (Registry | None): Registry for string / ``name`` forms.
            ``None`` is valid when using only ``_target_`` or plain values.
        **injected (Any): Caller-forced defaults for the top-level object
            (overridable by explicit params in the spec).

    Returns:
        T | Any: The constructed component, or ``spec`` unchanged for scalars.

    Raises:
        ValueError: If a mapping spec has no ``name`` or ``_target_`` key, or if
            a ``name`` spec is used without a registry.
    """
    if isinstance(spec, str):
        if registry is not None:
            return registry.create(spec, **injected)
        return spec  # plain string value inside a nested _target_ spec

    if isinstance(spec, Mapping):
        params = {k: v for k, v in spec.items() if k not in _HYDRA_META}
        target = params.pop("_target_", None)

        if target is not None:
            factory = resolve_target(str(target))
            resolved = {k: instantiate(v) for k, v in params.items()}
            return factory(**{**injected, **resolved})

        name = params.pop("name", None)
        if name is None:
            raise ValueError(f"Brick spec mapping needs a 'name' or '_target_' key: {dict(spec)!r}.")
        if registry is None:
            raise ValueError(f"A registry is required to resolve the name spec {name!r}.")
        return registry.create(str(name), **{**injected, **params})

    if isinstance(spec, list):
        return [instantiate(item) for item in spec]

    return spec
