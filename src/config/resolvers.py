"""OmegaConf resolvers: expose canonical ``core/keys.py`` constants to YAML.

Registered as an import side-effect of :mod:`src.config`, so any path that
resolves a config (``main.py`` at runtime, Hydra ``compose`` in tests) can
reference a key token via ``${key:NAME}`` — e.g.
``monitor: ${key:LOSS}/val/${key:TOTAL}`` resolves to ``loss/val/total``.

This keeps the string token defined once, in Python (:mod:`src.core.keys`):
change ``LOSS``/``TOTAL`` there and both the logged keys and the YAML that
monitors them follow. OmegaConf is a config-boundary detail, so the resolver
lives in the config layer, not in ``core``.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from src.core import keys


def register_resolvers() -> None:
    """Register the ``${key:NAME}`` resolver. Idempotent (safe to call repeatedly)."""
    OmegaConf.register_new_resolver("key", lambda name: getattr(keys, name), replace=True)
