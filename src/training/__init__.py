"""How a batch becomes a loss: the algorithm boundary, and the plain multitask objective behind it.

Importing this package is what makes its names resolvable: `learner: standard` in a config finds one here.
"""

from __future__ import annotations

from src.training.base import Learner
from src.training.learner import StandardLearner

__all__ = ["Learner", "StandardLearner"]
