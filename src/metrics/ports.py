"""What the training loop asks of a task's metrics, and what one metric may say about its readings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from torch import nn

if TYPE_CHECKING:
    from src.core.entities import TaskOutput


class MetricSet(nn.Module, ABC):
    """A stateful collection of metrics for one task and stage.

    Accumulates over batches, computes at epoch end, then resets. Keys
    returned by ``compute`` and ``directions`` match.
    """

    @abstractmethod
    def update(self, predictions: TaskOutput, target: TaskOutput) -> None:
        """Accumulate one batch of activated predictions against targets.

        Both sides are whatever the task's shape is — a tensor, or a set of objects. A metric
        given a shape it cannot compare refuses by name.
        """

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Return computed values keyed by metric name."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated state."""

    @abstractmethod
    def directions(self) -> dict[str, bool | None]:
        """Return each metric's ``higher_is_better`` flag, ``None`` when directionless.

        Lets consumers (checkpoint monitors, progress displays) rank values
        without re-deriving semantics from metric names.
        """


@runtime_checkable
class MultiReadingMetric(Protocol):
    """A metric whose computed value is several named readings rather than one number.

    Structural: a consumer that needs the list (a checkpoint monitor asking which keys
    exist) reads it without the metric inheriting anything.
    """

    readings: tuple[str, ...]
