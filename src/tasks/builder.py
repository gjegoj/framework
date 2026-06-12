"""TaskBuilder: the Bridge that combines a topology and an objective into a Task.

Validates that the (topology, objective) combination is supported, then sizes
the head, builds the bricks and clones metrics per stage (each stage needs its
own accumulating state).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from src.core.entities import HeadSpec, Task
from src.core.enums import Stage
from src.core.instantiate import BrickSpec
from src.metrics.builders import MetricsSpec
from src.tasks.strategies.objective import ObjectiveStrategy
from src.tasks.strategies.topology import TopologyStrategy

DEFAULT_STAGES: tuple[Stage, ...] = (Stage.TRAIN, Stage.VAL, Stage.TEST)


class TaskBuilder:
    """Assembles a ``Task`` from a topology and an objective strategy.

    Parameters:
        topology (TopologyStrategy): The output-structure axis.
        objective (ObjectiveStrategy): The label-semantics axis.
    """

    def __init__(self, topology: TopologyStrategy, objective: ObjectiveStrategy) -> None:
        self._topology = topology
        self._objective = objective

    def build(
        self,
        name: str,
        num_classes: int,
        weight: float = 1.0,
        stages: Sequence[Stage] = DEFAULT_STAGES,
        loss_spec: BrickSpec | None = None,
        metrics_spec: MetricsSpec | None = None,
        head_override: str | dict[str, Any] | None = None,
        feature_key_override: str | None = None,
    ) -> Task:
        """Build a task; raise if the topology/objective combination is invalid.

        Parameters:
            name (str): Unique task name.
            num_classes (int): Class count (drives head size and metrics).
            weight (float): Weight in the aggregated loss.
            stages (Sequence[Stage]): Stages to build metric sets for.
            loss_spec (BrickSpec | None): YAML ``loss:`` override; ``None`` -> objective default.
            metrics_spec (MetricsSpec | None): YAML ``metrics:`` override; ``None`` -> default.
            head_override: Optional head spec from task config.
                ``None`` → backbone native (prefer_native=True).
                ``str`` → registry key override (prefer_native=False).
                ``dict`` → ``{kind?, _target_?, ...options}`` override.
            feature_key_override (str | None): Override the topology-default
                stream key (e.g. ``"encoder_last"`` for smp classification).
                Applied before ``head_override`` so the key is visible to it.

        Returns:
            Task: The assembled task bundle.

        Raises:
            ValueError: If the objective is not supported on the topology.
        """
        if not self._objective.supports(self._topology.kind):
            raise ValueError(
                f"Objective {self._objective.kind.value!r} is not supported on topology {self._topology.kind.value!r}."
            )

        out_features = self._objective.out_features(num_classes)
        head_spec = self._topology.head_spec(out_features)
        if feature_key_override is not None:
            head_spec = dataclasses.replace(head_spec, feature_key=feature_key_override)
        if head_override is not None:
            head_spec = _apply_head_override(head_spec, head_override)
        metrics = {stage: self._objective.build_metrics(num_classes, metrics_spec) for stage in stages}
        return Task(
            name=name,
            head_spec=head_spec,
            codec=self._objective.build_task_codec(),
            criterion=self._objective.build_criterion(loss_spec),
            activation=self._objective.build_activation(),
            metrics=metrics,
            topology=self._topology.kind,
            objective=self._objective.kind,
            weight=weight,
        )


def _apply_head_override(base: HeadSpec, override: str | dict[str, Any]) -> HeadSpec:
    """Merge an explicit head override from task config into the topology's default spec."""
    if isinstance(override, str):
        return HeadSpec(
            kind=override,
            out_features=base.out_features,
            feature_key=base.feature_key,
            prefer_native=False,
        )
    params = dict(override)
    target = params.pop("_target_", None)
    kind = str(params.pop("kind", base.kind))
    return HeadSpec(
        kind=kind,
        out_features=base.out_features,
        feature_key=base.feature_key,
        options=params,
        prefer_native=False,
        target=target,
    )
