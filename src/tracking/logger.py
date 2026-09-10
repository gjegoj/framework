"""A single route for experiment parameters, scalar metrics and file artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path


class Logger(ABC):
    @abstractmethod
    def log_parameters(self, values: Mapping[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, values: Mapping[str, float], step: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, path: Path, metadata: Mapping[str, object]) -> None:
        raise NotImplementedError
