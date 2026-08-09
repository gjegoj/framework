"""Registries of the training capability: optimizers, schedules and profilers by name."""

from __future__ import annotations

from lightning.pytorch.profilers import AdvancedProfiler, Profiler, PyTorchProfiler, SimpleProfiler
from torch import optim

from src.core.registry import Registry

optimizer_registry: Registry[optim.Optimizer] = Registry("optimizer")
"""Config-facing optimizers."""

optimizer_registry.register("adamw")(optim.AdamW)
optimizer_registry.register("adam")(optim.Adam)
optimizer_registry.register("sgd")(optim.SGD)

scheduler_registry: Registry[optim.lr_scheduler.LRScheduler] = Registry("scheduler")
"""Config-facing learning-rate schedulers."""

scheduler_registry.register("cosine")(optim.lr_scheduler.CosineAnnealingLR)
scheduler_registry.register("onecycle")(optim.lr_scheduler.OneCycleLR)
scheduler_registry.register("plateau")(optim.lr_scheduler.ReduceLROnPlateau)
scheduler_registry.register("step")(optim.lr_scheduler.StepLR)

profiler_registry: Registry[Profiler] = Registry("profiler")
"""Config-facing profilers: where a run's wall clock went.

The three that run on the hardware this framework targets. Lightning knows a
fourth alias, ``xla``, which is left out on purpose — constructing it without
``torch_xla`` installed raises, and an alias that cannot be built is worse than
no alias, since the failure names a missing dependency rather than a wrong
choice. A TPU run reaches it by ``_target_`` like anything else.
"""

profiler_registry.register("simple")(SimpleProfiler)
profiler_registry.register("advanced")(AdvancedProfiler)
profiler_registry.register("pytorch")(PyTorchProfiler)
