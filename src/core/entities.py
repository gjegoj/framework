"""Core domain entities that flow through the training pipeline.

Plain dataclasses with no framework dependency beyond ``torch`` as the numerical
"language". They are the stable contract every layer reads/writes: the data
layer produces ``Sample``/``Batch``; the model produces ``FeatureBundle`` and a
``ModelOutput``; criteria produce ``LossResult``; steps produce ``StepOutput``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeGuard

import torch.nn as nn
from torch import Tensor

from src.core.keys import POOLED

if TYPE_CHECKING:
    from collections.abc import KeysView

    import torch

    from src.core.enums import Stage
    from src.core.ports import Activation, Criterion, MetricSet, TaskCodec
    from src.tasks.taxonomy import Objective, Topology


@dataclass
class Sample:
    """A single, un-batched example produced by the data layer.

    Values are intentionally loose (numpy arrays, tensors, scalars): a sample is
    assembled before collation and may carry several input modalities and
    several task targets.

    Parameters:
        inputs (dict[str, Any]): Named model inputs (e.g. ``{"image": ndarray}``).
        targets (dict[str, Any]): Named task targets keyed by target column.
        meta (dict[str, Any]): Free-form metadata (paths, source, ...).
    """

    inputs: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """A collated batch of samples ready for the model.

    Parameters:
        inputs (dict[str, Tensor]): Batched, named model inputs.
        targets (dict[str, Tensor]): Batched, named task targets.
        meta (dict[str, Any]): Aggregated per-sample metadata.
    """

    inputs: dict[str, Tensor] = field(default_factory=dict)
    targets: dict[str, Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> Batch:
        """Return a copy with all input/target tensors moved to ``device``.

        Implements the transferable-object protocol Lightning uses to move a
        batch onto the accelerator.

        Parameters:
            device (torch.device | str): Target device.

        Returns:
            Batch: New batch with tensors on ``device`` (metadata shared).
        """
        inputs = {key: value.to(device) for key, value in self.inputs.items()}
        targets = {key: value.to(device) for key, value in self.targets.items()}
        return Batch(inputs=inputs, targets=targets, meta=self.meta)


@dataclass
class FeatureBundle:
    """Named feature streams produced by a backbone.

    A backbone may expose several streams (``pooled`` for a global vector,
    ``decoder`` for a dense map, ``image_embed``/``text_embed`` for multimodal).
    Heads select the stream they consume by key.

    Parameters:
        streams (dict[str, Tensor]): Feature streams by name.
    """

    streams: dict[str, Tensor] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Tensor:
        try:
            return self.streams[key]
        except KeyError as error:
            available = sorted(self.streams)
            raise KeyError(f"Feature stream {key!r} not found. Available: {available}.") from error

    def __contains__(self, key: str) -> bool:
        return key in self.streams

    def keys(self) -> KeysView[str]:
        """Return the available stream names."""
        return self.streams.keys()


@dataclass
class LossResult:
    """Output of a criterion: a backprop scalar plus named components.

    Parameters:
        total (Tensor): Scalar loss used for backpropagation.
        components (dict[str, Tensor]): Individual named loss terms, for logging.
    """

    total: Tensor
    components: dict[str, Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """Result of a ``CompositeModel`` forward pass.

    Parameters:
        features (FeatureBundle): Raw backbone feature streams.
        task_logits (dict[str, Tensor]): Per-task head outputs (logits) by task name.
    """

    features: FeatureBundle
    task_logits: dict[str, Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class HeadSpec:
    """Declarative instruction for building a task head.

    Produced by the task layer (from topology + resolved class count) and
    consumed by model assembly, which sizes the head from the backbone's feature
    dimension and constructs it via the head registry.

    Three build modes (evaluated in order in ``build_composite_model``):

    1. **``target`` set** — full custom via ``_target_``.  ``in_features`` and
       ``in_channels`` are injected as defaults; ``options`` can override.
    2. **``prefer_native=True``** — ask the backbone for its native head (e.g.
       smp's ``SegmentationHead``, timm's ``create_classifier``).  Falls back to
       the registry if the backbone returns ``None``.
    3. **Registry** — ``head_builders.create(kind, in_features=…, **options)``.

    Parameters:
        kind (str): Head registry key used in fallback / explicit override.
        out_features (int): Output dimension (class count / regression dim).
        feature_key (str): Which ``FeatureBundle`` stream the head consumes.
        options (dict[str, Any]): Extra constructor kwargs for the head builder.
        prefer_native (bool): Try the backbone's native head before the registry.
        target (str | None): Fully-qualified ``_target_`` import path for a
            completely custom head class.
        view_keys (tuple[str, ...] | None): For RANKING topology — the ordered
            input alias names that form the N views (e.g. ``("anchor",
            "positive", "negative")``).  ``None`` for all non-ranking tasks.
        stream_keys (tuple[str, ...] | None): For MULTISTREAM topology — the
            ordered ``FeatureBundle`` stream names (one per encoder, e.g.
            ``("image", "text")``) the extractor stacks into ``[B, N, D]``.
            ``None`` for all non-multistream tasks.
    """

    kind: str
    out_features: int
    feature_key: str = POOLED
    options: dict[str, Any] = field(default_factory=dict)
    prefer_native: bool = False
    target: str | None = None
    view_keys: tuple[str, ...] | None = None
    stream_keys: tuple[str, ...] | None = None


@dataclass
class TargetView:
    """A task target adapted for loss and metric computation.

    Keeping the two views separate lets augmentations differ between them later
    (e.g. MixUp produces soft loss targets but hard metric targets).

    Parameters:
        loss (Tensor): Target shaped/typed for the criterion.
        metric (Tensor): Target shaped/typed for the metric set.
    """

    loss: Tensor
    metric: Tensor


@dataclass
class TaskStepView:
    """Per-task view after a step — shared by metrics and visualization.

    Parameters:
        preds (Tensor): Post-activation predictions (same tensor passed to ``MetricSet.update``).
        metric_target (Tensor): The metric target from ``TaskCodec.adapt`` (``TargetView.metric``).
    """

    preds: Tensor
    metric_target: Tensor


class StepOutput(TypedDict):
    """Lightning step return type — dict contract for train/val/test steps and callbacks.

    Lightning extracts ``loss`` for backprop; the whole dict flows to
    ``on_*_batch_end(outputs, ...)``. ``task_views`` carries everything a
    visualization callback needs (post-activation preds + metric targets per
    task) — the raw ``ModelOutput`` is intentionally not returned, so no autograd
    graph is held past the step.
    """

    loss: Tensor
    task_views: dict[str, TaskStepView]


def is_training_step_output(outputs: object) -> TypeGuard[StepOutput]:
    """Return whether ``outputs`` matches the :class:`StepOutput` contract."""
    return (
        isinstance(outputs, dict)
        and "loss" in outputs
        and "task_views" in outputs
        and isinstance(outputs["task_views"], dict)
    )


@dataclass
class Task:
    """A unit of learning: the bundle of bricks for one head.

    Assembled by the task layer (topology x objective). The model holds the
    parametric head (by name); this object carries everything else needed to
    compute the task's loss and metrics during a step.

    Parameters:
        name (str): Unique task name; also the key into ``Batch.targets``.
        head_spec (HeadSpec): How to build/route this task's head.
        codec (TaskCodec): Adapts the raw target into a ``TargetView``.
        criterion (Criterion): Loss computed on logits.
        activation (Activation): Maps logits to predictions for metrics.
        metrics (dict[Stage, MetricSet]): Per-stage metric collections.
        topology (Topology): Output-structure axis (GLOBAL/DENSE/...) this task composes.
        objective (Objective): Label-semantics axis (multiclass/.../metric) this task composes.
        weight (float): Weight of this task in the aggregated loss.
        class_names (list[str] | None): Ordered class names (index → name) for
            per-class metric logging and confusion-matrix axis labels.
            ``None`` for tasks without a class vocabulary (regression, etc.).
    """

    name: str
    head_spec: HeadSpec
    codec: TaskCodec
    criterion: Criterion
    activation: Activation
    metrics: dict[Stage, MetricSet]
    topology: Topology
    objective: Objective
    weight: float = 1.0
    class_names: list[str] | None = None

    @property
    def feature_key(self) -> str:
        """The feature stream this task's head consumes."""
        return self.head_spec.feature_key


@dataclass(frozen=True)
class ExportRequest:
    """Format-neutral export invocation — adapters own format-specific details.

    Parameters:
        module (nn.Module): Traceable export wrapper (eval + cpu applied by pipeline).
        example_inputs (tuple[Tensor, ...]): Dummy tensors for tracing.
        path (Path): Destination file path (suffix set by the exporter).
        input_names (list[str]): Logical input tensor names for the serialized graph.
        output_names (list[str]): Logical output tensor names.
        options (dict[str, Any]): Format-specific options (e.g. ``opset_version``,
            ``dynamic_batch`` for ONNX). The sole carrier of per-format settings.
    """

    module: nn.Module
    example_inputs: tuple[Tensor, ...]
    path: Path
    input_names: list[str]
    output_names: list[str]
    options: dict[str, Any] = field(default_factory=dict)
