"""What a metric can say about itself beyond its value, which torchmetrics has no word for."""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from src.core import Stage


@runtime_checkable
class DeclaresStages(Protocol):
    """A reading that is meaningful in some stages and not others, and names which.

    Most measurements answer the same question whatever the model is doing, so most metrics say
    nothing here and are taken everywhere. A reading that accumulates across an epoch is the exception:
    what it accumulates over training is several states of a moving model, and the number it then
    reports is about the drift between them rather than about the model.

    A fact about the measurement rather than a knob on it: there is no run for which ranking a gallery
    built from two different encoders is the number its name claims, so there is nothing to choose.

    A ``ClassVar``, as ``ShapeAware`` and ``Produces`` are: what a metric is meaningful in is settled by
    which metric it is, so there is nothing an instance could answer differently.
    """

    read_on: ClassVar[frozenset[Stage]]
