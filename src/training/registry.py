"""Registries of optimizer and LR-scheduler classes selectable from YAML by ``name``.

These are third-party ``torch.optim`` / ``torch.optim.lr_scheduler`` classes (no
local abstraction to co-locate with), so the registries live in their own module
— mirroring ``metrics/registry.py``. Users register their own the same way, or
bypass the registry with a ``_target_`` spec.
"""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LRScheduler

from src.core.registry import Registry

optimizers: Registry[torch.optim.Optimizer] = Registry("optimizer")

optimizers.register("adamw")(torch.optim.AdamW)
optimizers.register("adam")(torch.optim.Adam)
optimizers.register("sgd")(torch.optim.SGD)
optimizers.register("rmsprop")(torch.optim.RMSprop)

schedulers: Registry[LRScheduler] = Registry("scheduler")

schedulers.register("cosine")(torch.optim.lr_scheduler.CosineAnnealingLR)
schedulers.register("onecycle")(torch.optim.lr_scheduler.OneCycleLR)
schedulers.register("plateau")(torch.optim.lr_scheduler.ReduceLROnPlateau)
schedulers.register("step")(torch.optim.lr_scheduler.StepLR)
