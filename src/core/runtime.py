"""Runtime values discovered during a run, and lazy references to them.

Replaces the fragile string mechanism ``${runtime_value:...}`` from the old
prototype with an explicit, typed object. ``RuntimeContext`` is populated at
well-defined lifecycle points (after data setup: ``num_classes``/sizes; after
trainer attach: steps/device); ``RuntimeValue`` lets specs declare a dependency
on such a value, resolved just before the dependent object is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RuntimeValue:
    """A lazy reference to a value known only at runtime.

    Parameters:
        key (str): Field on ``RuntimeContext`` to read (e.g. ``"num_classes"``, ``"total_steps"``).
        task (str | None): Task name for per-task values such as ``num_classes``.
    """

    key: str
    task: str | None = None


@dataclass
class RuntimeContext:
    """Mutable container of values discovered during a run.

    Populated in phases: data setup fills ``num_classes``/``dataset_sizes``;
    trainer attachment fills ``steps_per_epoch``/``total_steps``/``device``.

    Parameters:
        num_classes (dict[str, int]): Per-task class counts inferred from data.
        dataset_sizes (dict[str, int]): Per-stage dataset sizes.
        epochs (int | None): Configured number of epochs.
        steps_per_epoch (int | None): Optimizer steps per epoch.
        total_steps (int | None): Total optimizer steps over training.
        device (torch.device | None): Active compute device.
        world_size (int): Distributed world size.
    """

    num_classes: dict[str, int] = field(default_factory=dict)
    dataset_sizes: dict[str, int] = field(default_factory=dict)
    epochs: int | None = None
    steps_per_epoch: int | None = None
    total_steps: int | None = None
    device: torch.device | None = None
    world_size: int = 1

    def resolve(self, ref: RuntimeValue) -> Any:
        """Resolve a single ``RuntimeValue`` against this context.

        Parameters:
            ref (RuntimeValue): The lazy reference to resolve.

        Returns:
            Any: The populated runtime value.

        Raises:
            ValueError: If a per-task value lacks a task name, or the value is not populated yet.
            KeyError: If a per-task value is requested for an unknown task.
        """
        if ref.key == "num_classes":
            if ref.task is None:
                raise ValueError("RuntimeValue('num_classes') requires a task name.")
            try:
                return self.num_classes[ref.task]
            except KeyError as error:
                known = sorted(self.num_classes)
                raise KeyError(f"num_classes not inferred for task {ref.task!r}. Known: {known}.") from error

        value = getattr(self, ref.key)
        if value is None:
            raise ValueError(f"RuntimeContext.{ref.key} is not populated yet.")
        return value


def resolve_runtime(value: Any, ctx: RuntimeContext) -> Any:
    """Recursively replace ``RuntimeValue`` references inside ``value`` using ``ctx``.

    Walks dicts, lists and tuples; leaves other values untouched.

    Parameters:
        value (Any): A scalar, ``RuntimeValue``, or nested container possibly holding them.
        ctx (RuntimeContext): Populated runtime context.

    Returns:
        Any: ``value`` with every ``RuntimeValue`` resolved.
    """
    if isinstance(value, RuntimeValue):
        return ctx.resolve(value)
    if isinstance(value, dict):
        return {key: resolve_runtime(item, ctx) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(resolve_runtime(item, ctx) for item in value)
    return value
