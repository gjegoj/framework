"""Building a task's metric sets from their declarations and its kind's facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.instantiate import fill_signature, refuse_a_declared_fact, resolve_params, resolve_target
from src.core.entities import TaskFacts
from src.core.taxonomy import Stage
from src.metrics.adapters import WrappedMetricSet
from src.metrics.registry import metric_registry

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from torchmetrics import Metric

    from src.config import MetricConfig
    from src.metrics.ports import MetricSet
    from src.tasks import TaskKind


def build_metric_sets(
    kind: TaskKind, facts: TaskFacts | None = None, metrics: Mapping[str, MetricConfig] | None = None
) -> dict[Stage, MetricSet]:
    """One independent ``MetricSet`` per stage for a task.

    Each entry is keyed by the label it logs under and names the metric explicitly, so two
    flavours of one metric stand side by side (``f1_macro`` / ``f1_micro``). ``None`` builds
    an empty set. The kind's ``metric_kwargs`` reach each metric through ``fill_signature``
    (the exception of ADR-0004), so ``mae`` beside ``accuracy`` is not handed a ``task`` it
    would refuse.
    A fact written in the declaration as well is refused by name: the data already said it.
    """
    components = dict(metrics) if metrics is not None else {}
    kwargs = kind.metric_kwargs(facts if facts is not None else TaskFacts())
    for component in components.values():
        refuse_a_declared_fact(component, *kwargs)
    return {
        stage: WrappedMetricSet({label: _metric(component, kwargs) for label, component in components.items()})
        for stage in Stage
    }


def _metric(component: MetricConfig, kwargs: Mapping[str, Any]) -> Metric:
    factory = resolve_target(component, metric_registry)
    built: Metric = factory(**resolve_params(component), **fill_signature(factory, **kwargs))
    return built
