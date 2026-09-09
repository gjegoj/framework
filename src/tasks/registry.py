"""Registries of the tasks capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.tasks.kinds import TaskKind

task_kind_registry: Registry[TaskKind] = Registry("task kind")
"""The familiar kinds of task by the name config spells them; a kind of your own arrives by ``_target_``."""
