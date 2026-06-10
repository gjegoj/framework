"""Logger factory: maps ``LoggerConfig.kind`` to a Lightning Logger or ``False``.

``False`` is Lightning's sentinel for "disable all logging" — pass it directly
to ``Trainer(logger=False)``.

Extension point: add new logger backends by adding a branch here and the
corresponding adapter in ``src/loggers/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import lightning as L

if TYPE_CHECKING:
    from src.config.schema import ExperimentConfig


def build_logger(config: "ExperimentConfig") -> Union[L.pytorch.loggers.Logger, bool]:
    """Build the experiment logger from config.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        Logger | bool: A configured Lightning Logger, or ``False`` to disable.

    Raises:
        ValueError: If ``config.logger.kind`` is not a known backend.
    """
    kind = config.logger.kind
    if kind == "none":
        return False
    if kind == "clearml":
        from src.loggers.clearml import ClearMLLogger

        project = config.logger.project or config.project
        task = config.logger.task or config.run_name
        return ClearMLLogger(project_name=project, task_name=task)
    raise ValueError(f"Unknown logger kind: {kind!r}. Known kinds: none, clearml.")
