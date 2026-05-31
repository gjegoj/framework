"""Build a ``MetricSet`` from a YAML ``metrics:`` spec (or sensible defaults).

Objective strategies supply ``base_kwargs`` (the torchmetrics ``task`` plus
``num_classes``/``num_labels`` they require); the user spec adds metric choices
and per-metric params (``top_k``, ``average``, ...). Absent a spec, a single
``accuracy`` is built so the simple case stays zero-config.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchmetrics import Metric, MetricCollection

from src.core.instantiate import resolve_target
from src.core.ports import MetricSet
from src.metrics.adapter import TorchMetricsAdapter
from src.metrics.registry import metric_factories

MetricsSpec = Mapping[str, Mapping[str, Any] | None]

_CLASSIFICATION_DEFAULT: MetricsSpec = {"accuracy": None}


def build_metric_set(
    spec: MetricsSpec | None,
    base_kwargs: Mapping[str, Any],
    default_spec: MetricsSpec | None = None,
) -> MetricSet:
    """Assemble a per-stage metric set from a config spec.

    Each entry is keyed by a display label; its value is the metric's params.
    The actual metric is the label itself unless the params carry an explicit
    ``name`` (registry key) or ``_target_`` (import path). ``base_kwargs`` are
    merged in first so user params can still override them.

    Parameters:
        spec (MetricsSpec | None): ``{label: {params}}``; ``None`` -> use ``default_spec``.
        base_kwargs (Mapping[str, Any]): Objective-supplied defaults (task, num_classes).
        default_spec (MetricsSpec | None): Fallback when both ``spec`` and this are ``None``
            uses ``_CLASSIFICATION_DEFAULT`` (accuracy). Objectives pass their own default.

    Returns:
        MetricSet: A fresh metric set (new state) for one stage.
    """
    fallback = default_spec if default_spec is not None else _CLASSIFICATION_DEFAULT
    entries = spec if spec is not None else fallback
    metrics: dict[str, Metric | MetricCollection] = {}
    for label, raw_params in entries.items():
        params = dict(raw_params or {})
        target = params.pop("_target_", None)
        if target is not None:
            # Fully custom metric: the user owns all args (no base_kwargs injection).
            metrics[label] = resolve_target(str(target))(**params)
            continue
        name = str(params.pop("name", label))
        factory = metric_factories.get(name)
        metrics[label] = factory(**{**base_kwargs, **params})
    return TorchMetricsAdapter(MetricCollection(metrics))
