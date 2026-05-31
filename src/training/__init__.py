"""Training: Lightning wrappers (humble objects) over the framework's domain logic."""

from src.training.aggregator import WeightedSumAggregator
from src.training.datamodule import LitDataModule
from src.training.module import LitModule
from src.training.optimizer import OptimizerBuilder
from src.training.registry import optimizers

__all__ = [
    "LitDataModule",
    "LitModule",
    "OptimizerBuilder",
    "WeightedSumAggregator",
    "optimizers",
]
