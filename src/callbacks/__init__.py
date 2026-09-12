"""What runs alongside the loop: Lightning's own hooks and ours, under the names a declaration writes.

Importing this package is what makes those names resolvable: `callbacks: [{name: checkpoint}]` finds
one here. No second callback framework is introduced — a callback is a Lightning callback.
"""

from __future__ import annotations

from src.callbacks.anneal import Anneal
from src.callbacks.batch_transform import ApplyBatchTransform
from src.callbacks.dataset_summary import DatasetSummary
from src.callbacks.ema import EmaWeights
from src.callbacks.freeze import Freeze
from src.callbacks.metric_summary import MetricSummary
from src.callbacks.model_summary import TreeModelSummary
from src.callbacks.progress import MetricsProgressBar
from src.callbacks.samples import SampleGrid

__all__ = [
    "Anneal",
    "ApplyBatchTransform",
    "DatasetSummary",
    "EmaWeights",
    "Freeze",
    "MetricSummary",
    "MetricsProgressBar",
    "SampleGrid",
    "TreeModelSummary",
]
