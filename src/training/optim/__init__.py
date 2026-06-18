"""Optimizer + LR-scheduler builders and their registries (selectable by ``name``)."""

from src.training.optim.optimizer import OptimizerBuilder, ParamGroupSpec
from src.training.optim.registry import optimizers, schedulers
from src.training.optim.scheduler import SCHEDULER_LR_PARAMS, TRAINER_FACTS, SchedulerBuilder

__all__ = [
    "SCHEDULER_LR_PARAMS",
    "TRAINER_FACTS",
    "OptimizerBuilder",
    "ParamGroupSpec",
    "SchedulerBuilder",
    "optimizers",
    "schedulers",
]
