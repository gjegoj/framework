"""Logger adapters and the ``logger_builders`` registry.

Concrete adapters live in submodules and are imported lazily so optional dependencies
(clearml, wandb) are not required at import time. The config→logger dispatch is
``build_logger`` in ``composition/wiring/training.py``.
"""

from src.loggers.registry import logger_builders

__all__ = ["logger_builders"]
