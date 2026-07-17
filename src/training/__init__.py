"""Training: Lightning wrappers (humble objects) over the framework's domain logic."""

from src.training.aggregator import WeightedSumAggregator
from src.training.modules import DistillationLitModule, LitDataModule, LitModule
from src.training.optim import OptimizerBuilder, SchedulerBuilder

__all__ = [
    "DistillationLitModule",
    "LitDataModule",
    "LitModule",
    "OptimizerBuilder",
    "SchedulerBuilder",
    "WeightedSumAggregator",
]
