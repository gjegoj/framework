"""Models: backbone/head adapters and multi-head assembly.

Importing this package registers the built-in backbones and heads so they are
available by key in the registries.
"""

import src.models.backbones  # noqa: F401 — importing registers the built-in backbones
import src.models.heads  # noqa: F401 — importing registers the built-in heads
from src.models.assembly import CompositeModel, build_composite_model
from src.models.ensemble import TeacherEnsemble
from src.models.registry import backbones, head_builders

__all__ = [
    "CompositeModel",
    "TeacherEnsemble",
    "backbones",
    "build_composite_model",
    "head_builders",
]
