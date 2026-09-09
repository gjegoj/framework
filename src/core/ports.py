"""Behaviour contracts of the core, implemented by capability packages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from torch import Tensor, nn

if TYPE_CHECKING:
    from src.core.entities import (
        Batch,
        Features,
        Loss,
        Prediction,
        Sample,
        StepResult,
    )
    from src.core.taxonomy import Geometry


type SampleTransform = Callable[[Sample], Sample]
"""Transforms one loaded sample — the augmentation seam of the data pipeline.

Takes a whole ``Sample`` rather than a single array because geometric
augmentation is joint: the crop applied to an image must be the same crop
applied to its masks.

**May write into the sample it is given**, and both shipped implementations do —
unlike a batch transform (``src.transforms.BatchTransform``), which promises a new
``Batch``. The asymmetry is deliberate and worth stating rather than discovering: a
sample has exactly one owner, the worker that just loaded it, so copying per item
would buy nothing; a batch is written into by a callback while other readers hold it.
"""


@runtime_checkable
class GeometryAware(Protocol):
    """A sample transform that carries arrays by their geometry, and is told which.

    Which inputs and targets travel through an augmentation, and as what — an image, a
    mask, boxes — is derived from the loaders and encoders as the pipeline is built, never in
    config. A transform is built from its declaration first and bound to that geometry
    afterwards, through this port, so the binding is one explicit call rather than a value
    slipped into a constructor by name — and a wrapper (``MultiViewTransform``) can pass
    it down to the pipeline it nests, which the composition root never sees. Structural:
    a transform of your own that needs no geometry implements nothing.
    """

    def with_geometry(
        self,
        inputs: Mapping[str, Geometry],
        targets: Mapping[str, Geometry],
        auxiliary_inputs: Mapping[str, Geometry],
    ) -> SampleTransform:
        """The same transform bound to these arrays; ``NONE`` geometries are never offered."""
        ...


class Model(nn.Module, ABC):
    """The trainable unit the training loop consumes — however it is built inside.

    One contract for every family: composed backbone-plus-heads, a model that arrives whole,
    or a decorator over another model. Implementations branch on ``self.training``,
    never on a stage argument.
    """

    @abstractmethod
    def step(self, batch: Batch) -> StepResult:
        """Run one forward pass and return the loss plus the predictions.

        Serves train/val/test: a single forward produces both what backward
        needs and what metrics consume.
        """

    @abstractmethod
    def predict(self, batch: Batch) -> Prediction:
        """Inference-only forward; must not require ``batch.targets``."""

    def task_parameters(self, task_name: str) -> Iterable[nn.Parameter]:
        """Parameters belonging to one task's own components — its head, its criterion.

        What a per-task learning rate binds to. A family without per-task parts keeps this
        default, and a rate declared against it is then refused rather than silently ignored.
        """
        return ()

    def criterion_of(self, task_name: str) -> nn.Module | None:
        """The criterion this family composes for one task; ``None`` when it composes none.

        Asked here rather than read off an attribute so the answer follows the model wherever it
        is nested — a schedule moving a loss's number has to reach one task's criterion without
        knowing how the family is built. ``None`` is the honest answer from a model that arrives
        whole and owns its loss.

        Raises:
            LookupError: From a family that composes criteria but has none under this name.
        """
        return None

    @property
    def architecture(self) -> str:
        """What this model is, in one token a run can be found by in a tracker.

        The composite family answers from its backbone, a decorator from what it wraps; a
        model that arrives whole keeps this default and is filed under its own class name.
        """
        return type(self).__name__


class Backbone(nn.Module, ABC):
    """Encodes named model inputs into named feature streams."""

    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> Features:
        """Encode ``inputs`` into the feature streams heads consume."""

    def __call__(self, inputs: dict[str, Tensor]) -> Features:
        """Typed delegate to ``nn.Module.__call__``: hooks keep working, and the return
        type torch erases to ``Any`` is restored for every call site.
        """
        return cast("Features", super().__call__(inputs))

    @abstractmethod
    def feature_dims(self) -> Mapping[str, int]:
        """The channel dimension of every stream this backbone exposes, by name.

        A mapping rather than one stream at a time: an adapter knows its streams at
        construction, and a caller can ask what a backbone offers.
        """

    def feature_dim(self, stream: str) -> int:
        """The channel dimension of one stream — what a head is sized from.

        Raises:
            LookupError: When this backbone exposes no such stream, naming the ones it does.
        """
        offered = self.feature_dims()
        try:
            return offered[stream]
        except KeyError:
            names = ", ".join(f"'{name}'" for name in sorted(offered)) or "no streams"
            raise LookupError(f"{type(self).__name__} exposes {names}, requested '{stream}'.") from None

    @property
    def architecture(self) -> str:
        """What this backbone is, in one token a run can be filtered by.

        The class name is the default; a wrapper over a library answers for itself — measured,
        timm normalises ``resnet18.a1_in1k`` to ``resnet18`` while smp calls a Unet
        ``u-resnet34`` — and a composite backbone joins what it holds.
        """
        return type(self).__name__

    def pyramid(self) -> tuple[str, ...]:
        """The streams a detection head reads, in reading order; ``()`` when this backbone has none.

        A concrete default rather than a protocol, as ``native_head`` is: every backbone can
        answer, and most answer "none". The names are the backbone's own — ``p3, p4, p5``
        by convention, or whatever a custom net calls the levels it hands over.
        """
        return ()

    def native_head(
        self, streams: tuple[str, ...], in_features: int | tuple[int, ...], out_features: int
    ) -> nn.Module | None:
        """Return the architecture's own head for ``streams``, or ``None``.

        ``None`` means the framework builds its own head; the builder consults this when a
        task prefers the native head, or when the kind takes the native one by default.
        """
        return None


def one_stream(features: Tensor | Mapping[str, Tensor], *, head: str) -> Tensor:
    """The single stream a head reads, refused by name when it was handed several.

    A head is any ``nn.Module`` taking a stream — or the pyramid, as a mapping — and
    returning a task's raw logits; a single-stream head on a multi-stream kind is a
    declaration error, not a shape to guess at.
    """
    if isinstance(features, Tensor):
        return features
    raise TypeError(
        f"{head} reads one stream, but was handed {len(features)}: {', '.join(features)}. "
        f"A head over several streams is declared by its kind or its backbone."
    )


class Criterion(nn.Module, ABC):
    """Computes a task's ``Loss`` from raw logits and a loss-view target.

    Criteria operate on logits, never on activated outputs — activations are
    a metrics/inference concern (``Activation``), which keeps losses
    numerically stable.
    """

    @abstractmethod
    def forward(self, logits: Tensor, target: Tensor) -> Loss:
        """Compute the loss with its named components."""

    def __call__(self, logits: Tensor, target: Tensor) -> Loss:
        """Typed delegate to ``nn.Module.__call__``, so hooks run and the type survives."""
        return cast("Loss", super().__call__(logits, target))
