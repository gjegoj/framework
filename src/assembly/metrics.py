"""Building per-stage metric sets from task declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.assembly.instantiate import instantiate
from src.core.entities import TargetFacts
from src.core.taxonomy import Stage
from src.metrics import WrappedMetricSet
from src.metrics.registry import metric_registry
from src.tasks.registry import objective_registry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.config.tasks import MetricConfig
    from src.core.ports import MetricSet
    from src.core.taxonomy import Objective


def build_metric_sets(
    objective: Objective,
    facts: TargetFacts | None = None,
    metrics: Mapping[str, MetricConfig] | None = None,
) -> dict[Stage, MetricSet]:
    """Build one independent ``MetricSet`` per stage for a task.

    Each entry is keyed by the *label* the metric logs under, and its value
    names the metric explicitly — ``name`` (a registry key) or ``_target_`` (an
    import path) — which is what lets two flavours of one metric stand side by
    side (``f1_macro`` / ``f1_micro``). ``None`` builds an empty set: defaults
    are the preset's word, injected when the config loads — a task built
    without a kind declares its own judgment.

    The objective's ``metric_kwargs`` (task mode, class counts from ``facts``)
    are offered as derived values: a metric receives the ones it names, so
    ``mae`` beside ``accuracy`` is not handed a ``task`` it would refuse.
    """
    behaviour = objective_registry.create(objective)
    components = dict(metrics) if metrics is not None else {}
    offered = behaviour.metric_kwargs(facts if facts is not None else TargetFacts())
    return {
        stage: WrappedMetricSet(
            {label: instantiate(component, metric_registry, **offered) for label, component in components.items()}
        )
        for stage in Stage
    }
