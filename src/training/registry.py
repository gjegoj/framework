"""The names an `optimizer`, `scheduler` or `learner` declaration may write."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import optim

from src.core import Registry

if TYPE_CHECKING:
    from src.training.base import Learner

learner_registry: Registry[Learner] = Registry("learner")
"""What `learner` writes; register with ``@learner_registry.register("name")``."""

optimizer_registry: Registry[optim.Optimizer] = Registry("optimizer")
"""What `optimizer` writes: torch's own classes, under the names a data scientist already uses.

Registered here rather than by decorator, because an optimizer that torch already implements needs
nothing from us. The list is a convenience, not a gate — anything torch offers is one ``_target_`` away.
"""

optimizer_registry.register("adamw")(optim.AdamW)
optimizer_registry.register("adam")(optim.Adam)
optimizer_registry.register("sgd")(optim.SGD)

scheduler_registry: Registry[optim.lr_scheduler.LRScheduler] = Registry("scheduler")
"""What `scheduler` writes; the same reasoning as the optimizers above."""

scheduler_registry.register("cosine")(optim.lr_scheduler.CosineAnnealingLR)
scheduler_registry.register("onecycle")(optim.lr_scheduler.OneCycleLR)
scheduler_registry.register("plateau")(optim.lr_scheduler.ReduceLROnPlateau)
scheduler_registry.register("step")(optim.lr_scheduler.StepLR)
