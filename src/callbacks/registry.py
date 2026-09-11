"""The names a `callbacks` declaration may write."""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from src.core import Registry

callback_registry: Registry[Callback] = Registry("callback")
"""What each entry of `callbacks` writes, under the names a config uses for them.

Lightning's own are registered here as they come: a callback the library already implements needs
nothing from us, and every knob of it stays reachable from the declaration. Ours register beside
their own definition, and anything else is one ``_target_`` away.
"""

callback_registry.register("lr_monitor")(LearningRateMonitor)
callback_registry.register("checkpoint")(ModelCheckpoint)
