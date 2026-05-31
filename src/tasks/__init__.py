"""Tasks: compose a ``Task`` from the topology x objective axes, via presets.

Importing this package registers the built-in topology/objective strategies and
task presets so they are resolvable by key.
"""

from src.tasks.activations import IdentityActivation, SigmoidActivation, SoftmaxActivation
from src.tasks.builder import DEFAULT_STAGES, TaskBuilder
from src.tasks.codecs import MulticlassTaskCodec
from src.tasks.presets import classification, task_presets
from src.tasks.strategies.objective import MulticlassObjective, ObjectiveStrategy, objective_strategies
from src.tasks.strategies.topology import GlobalTopology, TopologyStrategy, topology_strategies
from src.tasks.taxonomy import Objective, Topology

__all__ = [
    "DEFAULT_STAGES",
    "GlobalTopology",
    "IdentityActivation",
    "MulticlassObjective",
    "MulticlassTaskCodec",
    "Objective",
    "ObjectiveStrategy",
    "SigmoidActivation",
    "SoftmaxActivation",
    "TaskBuilder",
    "Topology",
    "TopologyStrategy",
    "classification",
    "objective_strategies",
    "task_presets",
    "topology_strategies",
]
