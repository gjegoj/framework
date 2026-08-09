"""The composition root: config in, a running experiment out."""

from __future__ import annotations

from src.assembly.experiment import Experiment, assemble, run
from src.assembly.instantiate import instantiate, resolve_target

__all__ = ["Experiment", "assemble", "instantiate", "resolve_target", "run"]
