"""Network contracts and composition of ready modules, without model-library imports."""

from __future__ import annotations

from src.models.backbones.base import Backbone
from src.models.base import GenerativeModel, Model, ModelWithLoss
from src.models.composite import CompositeModel, HeadConnection, select_features

__all__ = [
    "Backbone",
    "CompositeModel",
    "GenerativeModel",
    "HeadConnection",
    "Model",
    "ModelWithLoss",
    "select_features",
]
