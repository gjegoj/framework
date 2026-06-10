"""Wiring helpers: pure functions mapping a validated config to runtime objects.

Split by layer (``data`` / ``model`` / ``tasks`` / ``training`` / ``callbacks``)
for cohesion; this package re-exports the public builders so existing
``from src.composition.wiring import ...`` imports keep working.

Two construction families coexist by design:

- **Typed config sections** (backbone, optimizer, data, logger) have dedicated
  builders — their schema is fixed and ``kind``/``name`` would clash with the
  brick-spec grammar, so they are not brick-specs.
- **Free-form brick-specs** (transforms, codecs, losses, callbacks, batch
  transforms) all go through ``instantiate`` — one grammar, one home for
  ``_target_``. Context-aware construction is a Strategy registry
  (``callback_builders``), never an ``if`` ladder.
"""

from src.composition.wiring.callbacks import build_callbacks, callback_builders
from src.composition.wiring.common import WiringContext, forward_extras
from src.composition.wiring.data import (
    build_data_module,
    build_data_source,
    build_staged_sources,
    build_transforms,
)
from src.composition.wiring.model import build_backbone
from src.composition.wiring.tasks import build_bindings, build_tasks
from src.composition.wiring.training import (
    build_lit_module,
    build_logger,
    build_optimizer_builder,
    build_task_lr_overrides,
)

__all__ = [
    "WiringContext",
    "build_backbone",
    "build_bindings",
    "build_callbacks",
    "build_data_module",
    "build_data_source",
    "build_lit_module",
    "build_logger",
    "build_optimizer_builder",
    "build_staged_sources",
    "build_task_lr_overrides",
    "build_tasks",
    "build_transforms",
    "callback_builders",
    "forward_extras",
]
