"""Registries of the visualization capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.visualization.annotators import AnnotationObjective, AnnotationTopology

annotation_objective_registry: Registry[AnnotationObjective] = Registry("annotation objective")
"""How each ``Objective`` reads predictions; METRIC is absent — it has none to show per sample."""

annotation_topology_registry: Registry[AnnotationTopology] = Registry("annotation topology")
"""How each ``Topology`` draws a reading; the stacked-view members are absent — nothing per sample."""
