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

    Each entry is keyed by the label it logs under and names the metric explicitly, so two
    flavours of one metric stand side by side (``f1_macro`` / ``f1_micro``). ``None`` builds
    an empty set: defaults are the preset's, injected when the config loads. The objective's
    ``metric_kwargs`` are offered as derived values; a metric receives the ones it names.
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
