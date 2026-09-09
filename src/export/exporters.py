"""What every deployment format must be able to do: write a graph, and read it back."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from src.export.deployable import DeployableModel

type Runnable = Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]]
"""Runs a written artifact the way its own format loads it back.

Positional, like the graph's own ``forward``: a backend that feeds by name
(onnxruntime) reads the names out of the file it just wrote, which also proves
they landed in the graph.
"""


class Exporter(ABC):
    """Writes a deployable graph to a file, and reads it back to prove it.

    ``load`` is abstract: a format nobody can read back leaves the export unprovable.
    Tolerances live here because they are knowledge of the format — an fp16 engine drifts
    where a traced graph does not.

    Parameters:
        atol (float): Absolute output error tolerated against the source model.
        rtol (float): Relative output error tolerated against the source model.
    """

    suffix: ClassVar[str]
    """The extension the format writes its artifact under, declared by each backend."""

    def __init__(self, atol: float = 1e-4, rtol: float = 1e-3) -> None:
        self.atol = atol
        self.rtol = rtol

    def artifact_path(self, destination: Path) -> Path:
        """Where the artifact goes: ``destination`` under this format's suffix, its directory made.

        Appended rather than ``with_suffix``, because a destination whose name carries a dot
        (``model.v2``) would lose it. Built here so the formats cannot drift on it — each had
        its own copy, and only one carried the reason.
        """
        path = destination.parent / f"{destination.name}.{self.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        """Write ``model`` at ``destination`` (given without a suffix); return the file written.

        Returning the path instead of declaring an extension keeps a format that
        writes more than one file (ONNX external data) honest about which of
        them is the artifact.
        """

    @abstractmethod
    def load(self, path: Path) -> Runnable:
        """Read back what ``export`` wrote, as a callable."""
