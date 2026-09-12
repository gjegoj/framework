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
    library is imported. The word itself needs no translating — ``Semantics`` is a ``StrEnum`` whose
    members are the library's own spellings — so only the name it arrives under changes. A task whose
    target is a number has no semantics and gets no such translation.

    Added to the run's facts rather than put in their place. Both are offered because the metrics of a
    run are not all the library's: ours say ``semantics``, as every other part of this framework does,
    and a reader's own metric may be about a fact torchmetrics has no word for at all — the centres a
    binned target stands for were unreachable while this returned a closed four. ``fill_signature``
    hands each constructor only what it names, so no metric sees a word it did not ask for.
    """
    if facts.get("semantics") is None:
        return dict(facts)
    classes = facts.get("num_classes")
    return {**facts, "task": facts["semantics"], "num_labels": classes}


def _component(declared: object) -> ComponentConfig:
    """One grammar out of the two a declaration arrives in: a task's own default, or a validated section."""
    if isinstance(declared, ComponentConfig):
        return declared
    if isinstance(declared, str | Mapping):
        return ComponentConfig.model_validate(declared)
    raise TypeError(f"A metric is declared as a name or a component; got {type(declared).__name__}.")


def _one(declared: ComponentConfig, facts: Mapping[str, Any]) -> Metric:
    built = instantiate_offering(declared, metric_registry, **facts)
    if not isinstance(built, Metric):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a Metric: a run asks it to "
            "accumulate over a stage and to answer with one reading at the end, and this answers neither."
        )
    return built
