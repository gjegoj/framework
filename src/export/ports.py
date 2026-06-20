"""Export port: the ``ModelExporter`` interface each format backend implements.

Concrete backends (ONNX / TorchScript / TensorRT) implement this and self-register
in the ``exporters`` registry; the generic verifier composes ``load``/``validate``
into a parity report. It lives in the export subsystem (a "detail kept outward"),
not in ``core`` — nothing in an inner layer depends on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor

    from src.export.entities import ExportRequest

    # A loaded exported artifact: callable, named inputs → ordered output tensors.
    # Implemented per format by adapters (onnxruntime, torch.jit, …) so the generic
    # verifier can run any exported artifact uniformly. Used only in annotations.
    RunnableModel = Callable[[dict[str, Tensor]], tuple[Tensor, ...]]


class ModelExporter(ABC):
    """Export a traceable ``nn.Module`` to a deployment file format."""

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this format (e.g. ``.onnx``)."""

    @abstractmethod
    def export(self, request: ExportRequest) -> None:
        """Serialize ``request.module`` to ``request.path``."""

    def load(self, path: Path) -> RunnableModel | None:
        """Load the written artifact as a callable, for numerical parity checks.

        Override in a backend that can run its own format (e.g. onnxruntime,
        ``torch.jit``). Returning ``None`` means the backend has no runner, so the
        generic verifier skips parity for it.
        """
        return None

    def validate(self, request: ExportRequest) -> dict[str, str]:
        """Run format-specific static checks; return ``{name: detail}``.

        ``detail`` is ``""`` when the check passed, or the failure message when it
        failed. The default has no checks (empty dict), so non-validating backends
        degrade gracefully.
        """
        return {}
