"""Registries of the models capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.core.ports import Backbone, Head, Model
    from src.models.adapters import Adapters

head_registry: Registry[Head] = Registry("head")
"""Config-facing heads; register with ``@head_registry.register("name")``."""

backbone_registry: Registry[Backbone] = Registry("backbone")
"""Config-facing backbones; register with ``@backbone_registry.register("name")``."""

vendor_model_registry: Registry[Model] = Registry("vendor model")
"""Config-facing model families that arrive whole — head, loss and decoding their own.

Separate from ``backbone_registry`` because the two answer different questions: a
backbone is a piece this framework composes with, a model is a family it delegates to.
A name found here rather than there is what tells assembly which of the two it is
looking at, and is the only place that decision is made.

**Three classes implement ``Model`` and one is here.** That is the general rule in
``core.registry`` — a registry holds what a declaration *names*, never what it **only**
implies — applied to this port. Worth spelling out, because ``Registry[Model]`` promises
more than the membership delivers:

- ``YoloModel(model_name, num_classes, ...)`` is named: everything it needs comes from the
  ``model:`` declaration and one derived fact. **Registered.**
- ``CompositeModel(backbone, components)`` is the composer — what ``model:`` naming a
  *backbone* implies. Naming it would name the default.
- ``DistilledModel(student, teachers, criterion)`` is the decorator — what ``distillation:``
  being present implies. Naming it would be a second way to say that.

So membership is not "is this a ``Model``" but "does a run name it" — which is a role, and
roles are not subtypes. The registry's own name is where that has to be said, because the
type cannot say it.
"""

adapter_registry: Registry[Adapters] = Registry("adapters")
"""Config-facing parameter-efficient techniques; register with ``@adapter_registry.register("name")``."""
