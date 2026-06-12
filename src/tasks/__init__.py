"""Tasks: compose a ``Task`` from the topology x objective axes, via presets.

Importing this package registers the built-in topology/objective strategies and
task presets so they are resolvable by key.
"""

import src.losses.angular  # noqa: F401 — registers arcface criterion
import src.losses.contrastive  # noqa: F401 — registers info_nce criterion
import src.losses.ranking  # noqa: F401 — registers triplet_margin / margin_ranking criteria
from src.tasks.activations import IdentityActivation, SigmoidActivation, SoftmaxActivation
from src.tasks.builder import DEFAULT_STAGES, TaskBuilder
from src.tasks.codecs import BinaryTaskCodec, ContinuousTaskCodec, MulticlassTaskCodec, MultilabelTaskCodec
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
from src.tasks.taxonomy import Objective, Topology

__all__ = [
    "DEFAULT_STAGES",
    "BinaryObjective",
    "BinaryTaskCodec",
    "ContinuousObjective",
    "ContinuousTaskCodec",
    "DenseTopology",
    "GlobalTopology",
    "IdentityActivation",
    "MetricObjective",
    "MultistreamTopology",
    "MulticlassObjective",
    "MulticlassTaskCodec",
    "MultilabelObjective",
    "MultilabelTaskCodec",
    "Objective",
    "ObjectiveStrategy",
    "RankingTopology",
    "SigmoidActivation",
    "SoftmaxActivation",
    "TaskBuilder",
    "Topology",
    "TopologyStrategy",
    "classification",
    "contrastive",
    "objective_strategies",
    "pairwise_ranking",
    "regression",
    "segmentation",
    "TaskPreset",
    "task_presets",
    "topology_strategies",
    "triplet",
]
