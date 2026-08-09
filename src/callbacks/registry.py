"""Registry of the callbacks a config may name."""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from src.callbacks.anneal import AnnealCriterion
from src.callbacks.batch_transform import ApplyBatchTransform
from src.callbacks.dataset_summary import DatasetSummary
from src.callbacks.ema import EmaModelCheckpoint, EmaWeights
from src.callbacks.freeze import Freeze
from src.callbacks.metric_summary import MetricSummary
from src.callbacks.model_summary import TreeModelSummary
from src.callbacks.progress import MetricsProgressBar
from src.callbacks.samples import SampleGrid
from src.core.registry import Registry

callback_registry: Registry[Callback] = Registry("callback")
"""Config-facing callbacks; register with ``@callback_registry.register("name")``."""

# Lightning's own, registered rather than wrapped: ``Callback`` is already the port,
# and a wrapper would only add a layer to translate across.
callback_registry.register("lr_monitor")(LearningRateMonitor)
callback_registry.register("checkpoint")(ModelCheckpoint)

callback_registry.register("anneal")(AnnealCriterion)
callback_registry.register("freeze")(Freeze)
callback_registry.register("batch_transform")(ApplyBatchTransform)
callback_registry.register("ema")(EmaWeights)
callback_registry.register("ema_checkpoint")(EmaModelCheckpoint)
callback_registry.register("dataset_summary")(DatasetSummary)
callback_registry.register("metric_summary")(MetricSummary)
callback_registry.register("model_summary")(TreeModelSummary)
callback_registry.register("progress")(MetricsProgressBar)
callback_registry.register("samples")(SampleGrid)
