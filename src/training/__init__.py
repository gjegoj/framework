"""How a batch becomes a loss, and the loop that carries it: the algorithm boundary and Lightning's side.

Importing this package is what makes its names resolvable: `learner: standard` in a config finds one here.
"""

from __future__ import annotations

from src.training.base import Learner
from src.training.checkpoints import load_checkpoint, restore_best_weights, shipped_weights
from src.training.data import TrainingData
from src.training.learner import StandardLearner
from src.training.module import TrainingModule

__all__ = [
    "Learner",
    "StandardLearner",
    "TrainingData",
    "TrainingModule",
    "load_checkpoint",
    "restore_best_weights",
    "shipped_weights",
]
