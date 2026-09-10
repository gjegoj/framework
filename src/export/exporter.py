"""Format adapters; the common export procedure owns numerical comparison."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from src.core import TensorTree


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    path: Path
    format: str
    files: tuple[Path, ...]


class Exporter(ABC):
    @abstractmethod
    def export(self, model: nn.Module, example_inputs: Mapping[str, TensorTree], destination: Path) -> ExportedArtifact:
        """Export selected raw outputs, or explicitly requested supported Task processing."""
        raise NotImplementedError

    @abstractmethod
    def load(self, artifact: ExportedArtifact) -> Callable[[Mapping[str, TensorTree]], TensorTree]:
        """Load the written graph for comparison on real examples."""
        raise NotImplementedError
