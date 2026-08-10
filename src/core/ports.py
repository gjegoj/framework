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

    One contract for every family: a composed backbone-plus-heads model, a
    vendor self-contained model (YOLO-style), or a decorator over another
    model (distillation). Implementations branch on train/eval mode via
    ``self.training`` — never on a stage argument.
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
        """Parameters belonging to one task's own bricks — its head, its criterion.

        What a per-task learning rate binds to. The composite family serves
        them from its per-task registrations; a family without per-task parts
        keeps this default, and a rate declared against it is then refused
        rather than silently ignored.
        """
        return ()

    def criterion_of(self, task_name: str) -> nn.Module | None:
        """The criterion this family composes for one task; ``None`` when it composes none.

        The sibling of ``task_parameters``, and there for the same reason: something
        outside the model — a schedule moving a loss's own number over the run — has to
        reach one task's brick without knowing how this family is built, or whether a
        decorator wraps it. Asked here, the answer follows the model wherever it is
        nested; read off an attribute, it stops working the first time something else
        holds the model.

        ``None`` is the honest answer from a vendor family, whose loss is internal and
        has no per-task part to hand over.

        Raises:
            LookupError: From a family that does compose criteria, when it has none
                under this name — naming the ones it has, as ``DataModule.dataset``
                does for stages.
        """
        return None

    @property
    def architecture(self) -> str:
        """What this model is, in one token a run can be found by in a tracker.

        The composite family answers from its backbone and a decorator from what
        it wraps, so a run is filed under the thing that learned rather than
        under the scaffolding around it. A vendor family keeps this default and
        is filed under its own class name, which is honest — there is nothing
        else it could mean.
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

        Declared as a mapping rather than answered one stream at a time, because the
        answer *is* a mapping: an adapter knows its streams at construction and the set
        does not vary per call. Five adapters wrote the same branch-and-refuse over it,
        with five spellings of one sentence, and a caller had no way to ask what a
        backbone offers at all — which is the other half of what a stream name is for.
        """

    def feature_dim(self, stream: str) -> int:
        """The channel dimension of one stream — what a head is sized from.

        Concrete over ``feature_dims``, so the refusal below is written once and reads
        the same whichever adapter is holding the streams.

        Raises:
            LookupError: When this backbone exposes no such stream, naming the ones it
                does — as ``DataModule.dataset`` does for stages.
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

        The class name is the honest default; a wrapper over a library overrides
        it. There is no one rule for how, which is exactly why each wrapper
        answers for itself — measured, timm normalises ``resnet18.a1_in1k`` to
        ``resnet18``, so its own answer beats the declaration, while smp calls a
        Unet ``u-resnet34``, so the declaration beats its answer. A composite
        backbone joins what it holds, which is the case a config interpolation
        cannot reach at all.
        """
        return type(self).__name__

    def native_head(self, stream: str, in_features: int, out_features: int) -> nn.Module | None:
        """Return the architecture's own head for ``stream``, or ``None``.

        Override in concrete backbones to expose the source library's head
        (smp's ``SegmentationHead``, timm's classifier). ``None`` means the
        framework builds its own head; the builder only consults this when a
        task explicitly prefers the native head.
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

    The contract every data pipeline implements — table-driven (``TableDataModule``),
    folder-driven, or vendor-native (a YOLO data pipeline). Consumers depend on this
    interface, never on a concrete pipeline.

    The one data port that lives in the core: it is the data ↔ training boundary and
    mentions nothing beyond core entities and torch. The ports that would drag an I/O
    library in — table sources, target encoders, trackers — stay with their packages.
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

        Raises ``LookupError``, naming the stages it does have, when it has none
        for this one. That is an answer rather than a failure: a pipeline may
        legitimately carry no test data — a YOLO descriptor often ships without
        it, and per-stage sources need not declare all three — so a consumer is
        expected to catch it and say what it does instead. Stated here because
        both consumers and implementations rely on it.
        """

    def statistics(self) -> DatasetStatistics:
        """What this pipeline is about to serve, for the report drawn before epoch one.

        Concrete with an empty default, exactly as ``collate`` below: a pipeline
        that can describe its data overrides this, and one that cannot — a
        vendor-native loader that never sees an annotation table — answers with
        nothing. A consumer therefore always has something to call and something to
        say, where an optional-capability Protocol would have made a whole pipeline
        vanish from the report with no line explaining which one.
        """
        return DatasetStatistics()

    @property
    def collate(self) -> Callable[[list[Sample]], Batch] | None:
        """How this pipeline's samples become one batch; ``None`` takes the default.

        Batching belongs to the data, not to the training loop: detection
        targets are ragged — one image carries three boxes and the next
        eleven — so a vendor pipeline stacks them its own way and says so here.
        ``None`` rather than the framework's own function because this package
        may not import the one that implements it.
        """
        return None


def require_stage[T](datasets: Mapping[Stage, T] | None, stage: Stage, owner: str) -> T:
    """One stage's dataset, or the two refusals :meth:`DataModule.dataset` documents.

    A free function rather than a template method on the port: making ``dataset``
    concrete over an abstract ``_stage_datasets`` would impose a dict-of-stages on every
    future pipeline, including a lazy or streaming one that holds no such dict. This
    imposes nothing — a pipeline shaped that way calls it, one shaped otherwise does not
    — while the two implementations that *are* shaped that way stop carrying
    byte-identical copies of a contract the port had already written down in prose.

    Parameters:
        datasets (Mapping[Stage, T] | None): What ``setup`` built, or ``None`` before it ran.
        stage (Stage): The stage being asked for.
        owner (str): The pipeline's own name, for the sentence that names what to call first.
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

        Both sides are whatever the task's shape is — a tensor for a per-sample or
        per-pixel task, a set of objects for a per-instance one. A metric given a shape
        it cannot compare refuses by name rather than failing inside its library.
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

    Structural, like every other optional capability here: a metric declares which
    readings it publishes and consumers that need the list — a checkpoint monitor asking
    which keys exist, before a run has computed anything — read it without inheriting
    anything. A metric returning a single number never mentions it.
    """

    readings: tuple[str, ...]


@runtime_checkable
class AwaitsPreview(Protocol):
    """Something that reads a step's preview, and says beforehand whether it wants this one.

    A preview is cheap to make and not cheap to hold: ``.detach()`` shares storage with
    the outputs, and Lightning keeps a step's return value alive through the optimizer
    step, so an unconditional preview pins the activated outputs across ``backward()``.
    Measured at 352 MB for a ``[16, 21, 512, 512]`` segmentation batch, on every step,
    for a page most runs never draw.

    Asking removes that: Lightning runs ``on_*_batch_start`` before the step, so a
    consumer knows whether this batch is its own while the step can still act on it, and
    a run with no such consumer builds nothing ever.

    Argument-free on purpose — the consumer owns the whole decision (stage, cadence,
    batch index, rank) and answers with the one bit the module needs.
    """

    @property
    def awaiting_preview(self) -> bool: ...


@runtime_checkable
class DeclaresMetricDirections(Protocol):
    """A training module that reports its metrics' optimization directions.

    Structural on purpose: consumers (a progress display) colour improvements
    without re-deriving semantics from metric names, and a module without it
    degrades gracefully. Keys match the logged scalar keys
    (``{stage}/{task}/{label}``); values are ``higher_is_better`` flags,
    ``None`` when directionless.
    """

    def metric_directions(self) -> dict[str, bool | None]: ...
