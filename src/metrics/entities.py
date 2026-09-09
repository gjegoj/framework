"""What a metric returns that is not a number: a curve, a matrix, a per-class vector.

Completed values, no behaviour: the training loop routes them to whichever logger can draw
the shape, and names the classes where an artifact left ``labels`` / ``series`` open. Not
frozen — measured: torchmetrics' ``apply_to_collection`` refuses a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(slots=True)
class PerClass:
    """Values a metric produced per class, with which classes they are about.

    Position is *not* the class index: COCO's ``map_per_class`` covers only the classes that
    appeared and says which in ``classes``; a dense reading has ``classes == arange(len(values))``.
    Mutable, as every artifact a metric returns is (see the module docstring).
    """

    values: Tensor
    classes: Tensor

    def __post_init__(self) -> None:
        if len(self.values) != len(self.classes):
            raise ValueError(
                f"PerClass needs one class per value, got {len(self.values)} values and {len(self.classes)} classes."
            )

    def pairs(self) -> list[tuple[int, Tensor]]:
        """Each class with its value, read as vectors whatever shape they arrived in.

        Measured on torchmetrics 1.9.0: ``compute`` squeezes every tensor a metric returns, so a
        reading about one class reaches a reader as two 0-d tensors. Read by position that was
        ``zip(0, ...)``, and a one-class detector could not log its per-class reading.
        """
        return list(zip(self.classes.reshape(-1).tolist(), self.values.reshape(-1), strict=True))


@dataclass(slots=True)
class Curve:
    """A curve metric's plotted lines, already oriented for drawing.

    PR and ROC share one geometry with opposite axes, so orientation is stated by the metric
    that knew it. One entry per class; a binary metric carries the positive class's line.
    ``series is None`` means the lines live in the task's class space and the router fills
    the names. Mutable, as every artifact a metric returns is.
    """

    x: tuple[Tensor, ...]
    y: tuple[Tensor, ...]
    xaxis: str
    yaxis: str
    positive_only: bool = False
    series: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(f"A curve needs x and y per line, got {len(self.x)} x and {len(self.y)} y.")
        if self.series is not None and len(self.series) != len(self.x):
            raise ValueError(f"A curve with {len(self.x)} lines cannot carry {len(self.series)} series names.")


@dataclass(slots=True)
class Matrix:
    """A drawable 2-D artifact, axes named by the metric that knew them.

    ``labels is None`` means the index space is the task's classes and the router fills the
    names. Mutable, as every artifact a metric returns is.
    """

    value: Tensor
    xaxis: str
    yaxis: str
    labels: tuple[str, ...] | None = None
