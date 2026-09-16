"""The names a model section may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry

if TYPE_CHECKING:
    from torch import nn

    from src.models.adapters import Adapter
    from src.models.base import Backbone, Model

model_registry: Registry[Model] = Registry("model")
"""Families that compose a backbone with per-task heads; a whole network is reached by ``_target_``."""

backbone_registry: Registry[Backbone] = Registry("backbone")
"""Feature extractors a composite reads; one of your own is reached by ``_target_``, or named here
as ``Registry`` describes — a decorator alone leaves the name unresolvable until the module runs."""

adapter_registry: Registry[Adapter] = Registry("adapter")
"""Families of parameters a run adds to a network it did not build; a declaration names one."""

head_registry: Registry[nn.Module] = Registry("head")
"""Heads built at ``(in_features, out_features)`` alone; one of your own is reached by ``_target_``,
or named here as ``Registry`` describes."""
