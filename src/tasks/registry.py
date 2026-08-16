"""Registries of the tasks capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.tasks.objectives import TaskObjective
    from src.tasks.topologies import TaskTopology

objective_registry: Registry[TaskObjective] = Registry("objective")
"""Behaviour per ``Objective`` member; METRIC joins with metric learning."""

topology_registry: Registry[TaskTopology] = Registry("topology")
"""Behaviour per ``OutputTopology`` member — three entries, one per output structure."""
