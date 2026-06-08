"""Logger adapters and the build_logger factory.

Import ``build_logger`` from here; concrete adapters are in submodules and
imported lazily so optional dependencies (clearml, wandb) are not required at
import time.
"""

from src.loggers.registry import build_logger

__all__ = ["build_logger"]
