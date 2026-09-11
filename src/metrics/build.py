"""A declaration becomes the set of metrics one task is judged by, sized by what its target settled."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torchmetrics import Metric, MetricCollection

from src.config import ComponentConfig
from src.config.instantiate import instantiate_offering
from src.metrics.registry import metric_registry


def build_metrics(declared: Mapping[str, object], facts: Mapping[str, Any]) -> MetricCollection:
    """Every metric one task is judged by, keyed by the label a report shows it under.

    The label belongs to the declaration rather than to the metric, so two readings of one metric stand
    side by side (``f1`` beside ``f1_per_class``). Facts reach a metric by signature: a vocabulary goes
    to the metrics that score against one and nowhere else, which is what lets ``mae`` stand next to
    ``accuracy`` in a multitask run. A fact restated in the declaration is refused by name.

    Which declaration this is — the run's, or the task's own default — is settled before the call, so
    this package needs to know nothing about tasks. One collection per task: how many stages it is kept
    for is the training module's business, and ``MetricCollection.clone`` gives each its own state.
    Metrics sharing that state (precision, recall and f1 over one confusion matrix) are updated once
    between them; torchmetrics groups them on its own.
    """
    spoken = _in_torchmetrics_dialect(facts)
    return MetricCollection({label: _one(_component(one), spoken) for label, one in declared.items()})


def _in_torchmetrics_dialect(facts: Mapping[str, Any]) -> dict[str, Any]:
    """The run's facts under the names torchmetrics gives them.

    torchmetrics calls the label semantics ``task`` and takes one vocabulary under two names, depending
    on it. Translated here because this is the package torchmetrics is quarantined to: a task states
    what its labels mean in the framework's own word, and each library's dialect is spoken where that
    library is imported. A task whose target is a number has no semantics and is offered none.
    """
    if facts.get("semantics") is None:
        return {}
    classes = facts.get("num_classes")
    return {"task": str(facts["semantics"]), "num_classes": classes, "num_labels": classes}


def _component(declared: object) -> ComponentConfig:
    """One grammar out of the two a declaration arrives in: a task's own default, or a validated section."""
    if isinstance(declared, ComponentConfig):
        return declared
    if isinstance(declared, str | Mapping):
        return ComponentConfig.model_validate(declared)
    raise TypeError(f"A metric is declared as a name or a component; got {type(declared).__name__}.")


def _one(declared: ComponentConfig, facts: Mapping[str, Any]) -> Metric:
    try:
        built: Metric = instantiate_offering(declared, metric_registry, **facts)
    except TypeError as error:
        offered = ", ".join(sorted(facts)) or "nothing"
        raise ValueError(
            f"{declared.spelled!r} needs more than this task settles about its target ({offered}): {error}"
        ) from error
    return built
