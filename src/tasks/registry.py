"""The kinds of task a run may declare."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core import Registry

if TYPE_CHECKING:
    from src.tasks.base import Task

task_registry: Registry[Task] = Registry("task kind")
"""What `tasks.<name>.kind` writes; a kind of your own is reached by ``_target_``, or named here
as ``Registry`` describes — a decorator alone leaves the name unresolvable until the module runs."""
