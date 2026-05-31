"""Objective strategies: the label semantics axis of task composition.

An objective owns the target codec, criterion, activation, output size and
metrics. It declares which topologies it supports so invalid combinations fail
with a clear message at build time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.instantiate import BrickSpec, instantiate
from src.core.ports import Activation, Criterion, MetricSet, TaskCodec
from src.core.registry import Registry
from src.losses.criterion import criteria
from src.metrics.builders import MetricsSpec, build_metric_set
from src.tasks.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from src.tasks.codecs import (
    BinaryTaskCodec,
    ContinuousTaskCodec,
    MulticlassTaskCodec,
    MultilabelTaskCodec,
)
from src.tasks.taxonomy import Objective, Topology

_REGRESSION_METRICS: MetricsSpec = {"mse": None, "mae": None}


class ObjectiveStrategy(ABC):
    """Produces the loss/metric/codec bricks for a given label semantics.

    Each strategy supplies *defaults* for its objective and applies optional
    user overrides (``loss``/``metrics`` specs from YAML) on top — so the simple
    case stays zero-config while customisation never needs new code.

    Class attributes:
        kind (Objective): The objective this strategy implements.
        supported_topologies (frozenset[Topology]): Valid topology pairings.
        default_loss (str): ``criteria`` registry key used when ``loss:`` is absent.
        default_codec (str): ``target_codecs`` registry key for the data-layer
            target decoder this objective's labels require.
    """

    kind: Objective
    supported_topologies: frozenset[Topology]
    default_loss: str
    default_codec: str

    def supports(self, topology: Topology) -> bool:
        """Whether this objective is valid on ``topology``."""
        return topology in self.supported_topologies

    @abstractmethod
    def out_features(self, num_classes: int) -> int:
        """Number of head output channels for ``num_classes``."""

    @abstractmethod
    def build_task_codec(self) -> TaskCodec:
        """Build the task-layer target codec."""

    @abstractmethod
    def build_activation(self) -> Activation:
        """Build the logits->predictions activation for metrics/inference."""

    @abstractmethod
    def metric_base_kwargs(self, num_classes: int) -> dict[str, object]:
        """torchmetrics kwargs every metric for this objective needs (task, num_classes)."""

    def build_criterion(self, spec: BrickSpec | None = None) -> Criterion:
        """Build the loss; ``spec`` (YAML ``loss:``) overrides the objective default."""
        return instantiate(spec if spec is not None else self.default_loss, criteria)

    def default_metrics_spec(self) -> MetricsSpec | None:
        """Default ``metrics:`` spec when the user doesn't override. ``None`` → accuracy."""
        return None

    def build_metrics(self, num_classes: int, spec: MetricsSpec | None = None) -> MetricSet:
        """Build a fresh metric set; ``spec`` (YAML ``metrics:``) overrides the default."""
        return build_metric_set(
            spec,
            base_kwargs=self.metric_base_kwargs(num_classes),
            default_spec=self.default_metrics_spec(),
        )


objective_strategies: Registry[ObjectiveStrategy] = Registry("objective")


@objective_strategies.register(Objective.MULTICLASS)
class MulticlassObjective(ObjectiveStrategy):
    """Mutually-exclusive classes: softmax + cross-entropy + accuracy."""

    kind = Objective.MULTICLASS
    supported_topologies = frozenset({Topology.GLOBAL, Topology.DENSE})
    default_loss = "cross_entropy"
    default_codec = "label_index"

    def out_features(self, num_classes: int) -> int:
        return num_classes

    def build_task_codec(self) -> TaskCodec:
        return MulticlassTaskCodec()

    def build_activation(self) -> Activation:
        return SoftmaxActivation()

    def metric_base_kwargs(self, num_classes: int) -> dict[str, object]:
        return {"task": "multiclass", "num_classes": num_classes}


@objective_strategies.register(Objective.BINARY)
class BinaryObjective(ObjectiveStrategy):
    """Single binary decision: sigmoid + BCE + binary accuracy/AUROC."""

    kind = Objective.BINARY
    supported_topologies = frozenset({Topology.GLOBAL, Topology.DENSE})
    default_loss = "bce"
    default_codec = "label_index"

    def out_features(self, num_classes: int) -> int:
        return 1

    def build_task_codec(self) -> TaskCodec:
        return BinaryTaskCodec()

    def build_activation(self) -> Activation:
        return SigmoidActivation()

    def metric_base_kwargs(self, num_classes: int) -> dict[str, object]:
        return {"task": "binary"}


@objective_strategies.register(Objective.MULTILABEL)
class MultilabelObjective(ObjectiveStrategy):
    """Independent per-class decisions: sigmoid + BCE + multilabel F1/mAP."""

    kind = Objective.MULTILABEL
    supported_topologies = frozenset({Topology.GLOBAL, Topology.DENSE})
    default_loss = "bce"
    default_codec = "multilabel_binarize"

    def out_features(self, num_classes: int) -> int:
        return num_classes

    def build_task_codec(self) -> TaskCodec:
        return MultilabelTaskCodec()

    def build_activation(self) -> Activation:
        return SigmoidActivation()

    def metric_base_kwargs(self, num_classes: int) -> dict[str, object]:
        return {"task": "multilabel", "num_labels": num_classes}


@objective_strategies.register(Objective.CONTINUOUS)
class ContinuousObjective(ObjectiveStrategy):
    """Scalar regression: identity + MSE + MSE/MAE metrics."""

    kind = Objective.CONTINUOUS
    supported_topologies = frozenset({Topology.GLOBAL, Topology.DENSE})
    default_loss = "mse"
    default_codec = "float"

    def out_features(self, num_classes: int) -> int:
        return num_classes  # num_classes carries dim for regression

    def build_task_codec(self) -> TaskCodec:
        return ContinuousTaskCodec()

    def build_activation(self) -> Activation:
        return IdentityActivation()

    def metric_base_kwargs(self, num_classes: int) -> dict[str, object]:
        return {}  # torchmetrics MSE/MAE need no task/num_classes

    def default_metrics_spec(self) -> MetricsSpec | None:
        return _REGRESSION_METRICS
