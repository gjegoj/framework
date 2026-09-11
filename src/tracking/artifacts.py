"""What a metric returns that is not a number: a value a tracker draws rather than plots.

Completed values, no behaviour. Deliberately not frozen: measured on torchmetrics 1.9.0, the wrapper
around every ``Metric.compute`` walks the returned value with ``apply_to_collection``, which refuses a
frozen dataclass outright.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class Matrix:
    """A two-dimensional reading with its axes named by the metric that knew them.

    Which axis holds the prediction cannot be read off the tensor, and a chart drawn the other way
    round is a plausible-looking lie, so the metric states it here rather than leaving it to be guessed.
    """

    value: Tensor
    xaxis: str
    yaxis: str

    def __post_init__(self) -> None:
        if self.value.ndim != 2:
            raise ValueError(f"A matrix is drawn from two axes; this reading has {self.value.ndim}.")
