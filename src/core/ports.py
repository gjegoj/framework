"""Behaviour contracts of the core, implemented by capability packages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from torch import nn

# At runtime, not under TYPE_CHECKING: `DataModule.statistics` builds one as its default.
from src.core.entities import DatasetStatistics

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset

    from src.core.entities import (
        AdaptedTarget,
        Batch,
        DataProfile,
        Features,
        Loss,
        Prediction,
        Sample,
        StepResult,
        TaskOutput,
    )
    from src.core.taxonomy import Stage

type Activation = Callable[[Tensor], Tensor]
"""Maps raw logits to predictions for metrics and inference — never for the loss."""

type TargetAdapter = Callable[[Tensor], AdaptedTarget]
"""Shapes one raw batched target into its loss and metric views."""

type SampleTransform = Callable[[Sample], Sample]
"""Transforms one loaded sample — the augmentation seam of the data pipeline.

Takes a whole ``Sample`` rather than a single array because geometric
augmentation is joint: the crop applied to an image must be the same crop
applied to its masks.

**May write into the sample it is given**, and both shipped implementations do —
unlike ``BatchTransform`` below, which promises a new ``Batch``. The asymmetry is
deliberate and worth stating rather than discovering: a sample has exactly one
owner, the worker that just loaded it, so copying per item would buy nothing;
a batch is written into by a callback while other readers hold it.
"""

type BatchTransform = Callable[[Batch], Batch]
"""Transforms one collated batch — the seam for augmentations that mix samples.

A ``SampleTransform`` cannot do this: while one sample is being loaded, the
samples it would mix with do not exist yet. Returns a new ``Batch`` rather than
mutating, because the callback that applies one is the single place a batch is
written into.
"""


class Model(nn.Module, ABC):
    """The trainable unit the training loop consumes — however it is built inside.

    One contract for every family: composed backbone-plus-heads, a vendor self-contained
    model, or a decorator over another model. Implementations branch on ``self.training``,
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
        knowing how the family is built. ``None`` is the honest answer from a vendor family.

        Raises:
            LookupError: From a family that composes criteria but has none under this name.
        """
        return None

    @property
    def architecture(self) -> str:
        """What this model is, in one token a run can be found by in a tracker.

        The composite family answers from its backbone, a decorator from what it wraps; a
        vendor family keeps this default and is filed under its own class name.
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

    def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
        """Return the architecture's own head for ``stream``, or ``None``.

        ``None`` means the framework builds its own head; the builder consults this only when
        a task prefers the native head.
        """
        return None


class Head(nn.Module, ABC):
    """Maps one feature stream to a task's raw logits (pre-activation)."""

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Project ``features`` to logits for one task."""

    def __call__(self, features: Tensor) -> Tensor:
        """Typed delegate to ``nn.Module.__call__``, so hooks run and the type survives."""
        return cast("Tensor", super().__call__(features))


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


class DataModule(ABC):
    """The data side of an experiment: per-stage datasets plus inferred facts.

    The one data port in the core, because it is the data ↔ training boundary; the ports
    that would drag an I/O library in (sources, encoders, trackers) stay with their packages.
    """

    @abstractmethod
    def setup(self, profile: DataProfile) -> None:
        """Prepare datasets and record inferred facts into ``profile``.

        Runs before tasks and heads are assembled — the ordering that lets
        output sizes come from data instead of config.
        """

    @abstractmethod
    def dataset(self, stage: Stage) -> Dataset[Sample]:
        """Return the dataset for ``stage``; ``setup`` must have run first.

        Raises ``LookupError`` naming the stages it does have when it has none for this one —
        an answer, not a failure: a pipeline may legitimately carry no test data, and the
        consumer says what it does instead.
        """

    def statistics(self) -> DatasetStatistics:
        """What this pipeline is about to serve, for the report drawn before epoch one.

        Concrete with an empty default, as ``collate`` is: a pipeline that cannot describe its
        data (a vendor-native loader) answers with nothing, and the report still names it.
        """
        return DatasetStatistics()

    @property
    def collate(self) -> Callable[[list[Sample]], Batch] | None:
        """How this pipeline's samples become one batch; ``None`` takes the default.

        Batching belongs to the data: detection targets are ragged, so a vendor pipeline stacks
        them its own way and says so here. ``None`` rather than the framework's own function
        because this package may not import the one that implements it.
        """
        return None


def require_stage[T](datasets: Mapping[Stage, T] | None, stage: Stage, owner: str) -> T:
    """One stage's dataset, or the two refusals :meth:`DataModule.dataset` documents.

    A free function rather than a template method, so a lazy or streaming pipeline that holds
    no dict of stages is not forced to have one.

    Parameters:
        datasets (Mapping[Stage, T] | None): What ``setup`` built, or ``None`` before it ran.
        stage (Stage): The stage being asked for.
        owner (str): The pipeline's own name, for the message.
    """
    if datasets is None:
        raise RuntimeError(f"{owner}.setup(profile) must run before requesting datasets.")
    try:
        return datasets[stage]
    except KeyError:
        available = ", ".join(datasets) or "none"
        raise LookupError(f"No dataset for stage '{stage}'. Available stages: {available}.") from None


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
