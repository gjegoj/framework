"""Callback registry — maps string keys to Lightning callback classes.

Built-in callbacks are pre-registered. Custom callbacks can be added via
``callback_registry.register("key")(MyCallback)`` or via the ``_target_``
escape hatch in YAML (no registration needed).
"""

from __future__ import annotations

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from src.callbacks.batch_transform import BatchTransformCallback
from src.callbacks.ema import EmaCallback
from src.callbacks.freeze import FreezeCallback
from src.callbacks.progress_bar import MetricsProgressBar
from src.callbacks.sample_log import SampleLogCallback
from src.core.registry import Registry

callback_registry: Registry = Registry("callback")

callback_registry.register("lr_monitor")(LearningRateMonitor)
callback_registry.register("ema")(EmaCallback)
callback_registry.register("checkpoint")(ModelCheckpoint)
callback_registry.register("freeze")(FreezeCallback)
callback_registry.register("progress_bar")(MetricsProgressBar)
callback_registry.register("batch_transform")(BatchTransformCallback)
callback_registry.register("sample_log")(SampleLogCallback)
