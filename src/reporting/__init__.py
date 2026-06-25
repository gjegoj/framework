"""Dataset-distribution reporting package.

Provides a registry-based renderer pipeline that maps each distribution type to
a terminal table and an experiment-logger call without any ``isinstance``
branching on distribution type in the dispatch path.

Public API:
    - ``report_dataset_statistics``: print tables + log plots for all tasks.
    - ``DistributionRenderer``: ABC for adding new distribution renderers.
    - ``distribution_renderers``: registry keyed by distribution type ``__name__``.
"""

from src.reporting.renderers import (
    CategoricalDistributionRenderer,
    ContinuousDistributionRenderer,
    DistributionRenderer,
    distribution_renderers,
)
from src.reporting.report import report_dataset_statistics

__all__ = [
    "report_dataset_statistics",
    "DistributionRenderer",
    "distribution_renderers",
    "CategoricalDistributionRenderer",
    "ContinuousDistributionRenderer",
]
