"""Callbacks: Lightning callbacks (EMA, freeze, checkpoint, progress bar, sample log, batch transform).

Importing this package registers the built-in callbacks in ``callback_registry``.
"""

from src.callbacks.batch_transform import BatchTransformCallback
from src.callbacks.dataset_stats import DatasetStatsCallback
from src.callbacks.ema import EmaCallback
from src.callbacks.freeze import FreezeCallback
from src.callbacks.metric_summary import MetricSummaryCallback
from src.callbacks.model_summary import TreeModelSummary
from src.callbacks.progress_bar import MetricsProgressBar
from src.callbacks.registry import callback_registry
from src.callbacks.sample_log import SampleLogCallback

__all__ = [
    "BatchTransformCallback",
    "DatasetStatsCallback",
    "EmaCallback",
    "FreezeCallback",
    "MetricSummaryCallback",
    "MetricsProgressBar",
    "SampleLogCallback",
    "TreeModelSummary",
    "callback_registry",
]
