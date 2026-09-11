"""The names a `tracker` declaration may write."""

from __future__ import annotations

from lightning.pytorch.loggers import CSVLogger, Logger

from src.core import Registry

tracker_registry: Registry[Logger] = Registry("tracker")
"""What `tracker` writes: where a run's numbers are recorded.

Lightning's own CSV logger is registered here as it comes, under the name a config writes — a backend
the library already implements needs nothing from us, exactly as with the torch optimizers. A backend
of ours registers beside its own definition, and anything else is one ``_target_`` away.
"""

tracker_registry.register("csv")(CSVLogger)
