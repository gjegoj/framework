"""Future display-library adapter; callback owns selection and reporting schedule."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

from src.core import Batch, DatasetInfo, Prediction


class SampleVisualizer(ABC):
    @abstractmethod
    def render(
        self, batch: Batch, predictions: Prediction, targets: Mapping[str, object], info: DatasetInfo, destination: Path
    ) -> Path:
        """Convert selected samples to CPU/display types on demand, without model execution."""
        raise NotImplementedError
