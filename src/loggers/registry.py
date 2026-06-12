"""Logger registry: maps ``LoggerConfig.kind`` to a Lightning Logger or ``False``.

``False`` is Lightning's sentinel for "disable all logging" — pass it directly
to ``Trainer(logger=False)``.

Extension point (per the framework's "extension points are registries" rule):
register a new backend with ``@logger_builders.register("kind")`` instead of
editing ``build_logger``. Each builder takes the validated ``ExperimentConfig``
so it can resolve context (project/task names) and import its adapter lazily —
mirroring ``callback_builders`` in ``composition/wiring/callbacks.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L

from src.core.registry import Registry

if TYPE_CHECKING:
    from src.config.schema import ExperimentConfig

logger_builders: Registry[L.pytorch.loggers.Logger | bool] = Registry("logger")


@logger_builders.register("none")
def _build_none(config: ExperimentConfig) -> L.pytorch.loggers.Logger | bool:
    return False


@logger_builders.register("clearml")
def _build_clearml(config: ExperimentConfig) -> L.pytorch.loggers.Logger | bool:
    from src.loggers.clearml import ClearMLLogger

    project = config.logger.project or config.project
    task = config.logger.task or config.run_name
    return ClearMLLogger(project_name=project, task_name=task, tags=config.logger.tags)


def build_logger(config: ExperimentConfig) -> L.pytorch.loggers.Logger | bool:
    """Build the experiment logger from config.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        Logger | bool: A configured Lightning Logger, or ``False`` to disable.

    Raises:
        ValueError: If ``config.logger.kind`` is not a registered backend.
    """
    kind = config.logger.kind
    if kind not in logger_builders:
        known = ", ".join(sorted(logger_builders.keys()))
        raise ValueError(f"Unknown logger kind: {kind!r}. Known kinds: {known}.")
    return logger_builders.create(kind, config)
