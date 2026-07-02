"""Tasks: compose a ``Task`` from the topology x objective axes, via presets.

Importing this package registers the built-in topology/objective strategies and
task presets so they are resolvable by key.
"""

from src.core.taxonomy import Objective, Topology
from src.tasks.activations import IdentityActivation, SigmoidActivation, SoftmaxActivation
from src.tasks.adapters import (
    BinaryTargetAdapter,
    ContinuousTargetAdapter,
    MetricTargetAdapter,
    MulticlassTargetAdapter,
    MultilabelTargetAdapter,
)
from src.tasks.builder import DEFAULT_STAGES, TaskBuilder
from src.tasks.presets import (
    TaskPreset,
    classification,
    contrastive,
    pairwise_ranking,
    regression,
    segmentation,
    task_presets,
    triplet,
)
from src.tasks.strategies.objective import (
    BinaryObjective,
    ContinuousObjective,
    MetricObjective,
    MulticlassObjective,
    MultilabelObjective,
    ObjectiveStrategy,
    objective_strategies,
)
from src.tasks.strategies.topology import (
    DenseTopology,
    GlobalTopology,
    MultistreamTopology,
    RankingTopology,
    TopologyStrategy,
    topology_strategies,
)

__all__ = [
    "BinaryObjective",
    "BinaryTargetAdapter",
    "ContinuousObjective",
    "ContinuousTargetAdapter",
    "DEFAULT_STAGES",
    "DenseTopology",
    "GlobalTopology",
    "IdentityActivation",
    "MetricObjective",
    "MetricTargetAdapter",
    "MulticlassObjective",
    "MulticlassTargetAdapter",
    "MultilabelObjective",
    "MultilabelTargetAdapter",
    "MultistreamTopology",
    "Objective",
    "ObjectiveStrategy",
    "RankingTopology",
    "SigmoidActivation",
    "SoftmaxActivation",
    "TaskBuilder",
    "TaskPreset",
    "Topology",
    "TopologyStrategy",
    "classification",
    "contrastive",
    "objective_strategies",
    "pairwise_ranking",
    "regression",
    "segmentation",
    "task_presets",
    "topology_strategies",
    "triplet",
]
