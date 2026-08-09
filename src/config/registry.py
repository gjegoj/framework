"""Registries of the config surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.config.presets import TaskPreset

task_preset_registry: Registry[TaskPreset] = Registry("task preset")
"""Familiar task names as kinds of task; resolved via ``resolve_preset``.

A preset names a *kind of task* — its point on the axes and, when the kind is judged
differently than its label semantics alone would suggest, its customary metrics. Two
presets may share a point; none may drift off one.

Populated in ``presets.py``, the table beside the model it registers — the same split
every capability keeps between its ``registry.py`` and what registers into it.
"""
