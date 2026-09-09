"""Registries of the visualization capability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.visualization.renderers import LabelRenderer, MediaRenderer

label_renderer_registry: Registry[LabelRenderer[Any]] = Registry("label renderer")
"""One renderer per kind of label, keyed by the entity type itself.

The keys are types, not config names — a renderer is chosen by what the
annotator produced, never by a declaration. Same mechanism as every other
registry; the ``{name: ...}`` sugar is simply never poured over this one.
"""

media_renderer_registry: Registry[MediaRenderer[Any]] = Registry("media renderer")
"""One renderer per kind of media, keyed the same way."""
