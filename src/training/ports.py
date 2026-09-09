"""The handshake between the training module and its callbacks: what a step hands back, and what a callback may ask.

A Lightning hook contract, not a domain entity: ``LightningStepOutput`` is the dict Lightning
passes to ``on_*_batch_end`` verbatim, ``StepPreview`` the slice of a step a drawing callback
reads from it, and the two Protocols are what a module answers structurally — a callback asks
``isinstance`` rather than knowing the module's class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, NotRequired, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

    from src.core.entities import TaskOutput


class LightningStepOutput(TypedDict):
    """What a training step hands back — Lightning's own contract.

    ``loss`` is back-propagated. ``preview`` reaches every ``on_*_batch_end`` hook because
    Lightning passes the return value there verbatim; it is ``NotRequired`` because a preview
    is built only when an ``AwaitsPreview`` asked for this batch — holding one keeps the
    activated outputs alive through the optimizer step.
    """

    loss: Tensor
    preview: NotRequired[StepPreview]


@dataclass(frozen=True, slots=True)
class StepPreview:
    """What a step produced, detached — enough to draw it, nothing that holds a graph.

    Not the ``StepResult`` itself: that would carry the loss's ``grad_fn`` and every feature
    stream. Measured: 352 MB of outputs for a ``[16, 21, 512, 512]`` segmentation batch.
    """

    KEY: ClassVar[str] = "preview"
    """The key it is stored under in a step's return value; the writer and the reader agree here."""

    outputs: dict[str, TaskOutput]
    targets: dict[str, TaskOutput]


def preview_of(step_output: object) -> StepPreview | None:
    """The preview a step returned, or ``None`` when the module returned something else.

    Typed here rather than at each call site: Lightning types a hook's ``outputs`` as
    ``Tensor | Mapping | None``.
    """
    if isinstance(step_output, Mapping):
        preview = step_output.get(StepPreview.KEY)
        if isinstance(preview, StepPreview):
            return preview
    return None


@runtime_checkable
class AwaitsPreview(Protocol):
    """Something that reads a step's preview, and says beforehand whether it wants this one.

    Lightning keeps a step's return value alive through the optimizer step, so an
    unconditional preview pins the activated outputs across ``backward()`` — measured: 352 MB
    for a ``[16, 21, 512, 512]`` segmentation batch. Asked in ``on_*_batch_start``, a run with
    no consumer builds nothing. Argument-free: the consumer owns the whole decision.
    """

    @property
    def awaiting_preview(self) -> bool: ...


@runtime_checkable
class DeclaresMetricDirections(Protocol):
    """A training module that reports its metrics' optimization directions.

    Keys match the logged scalar keys (``{stage}/{task}/{label}``); values are
    ``higher_is_better`` flags, ``None`` when directionless.
    """

    def metric_directions(self) -> dict[str, bool | None]: ...
