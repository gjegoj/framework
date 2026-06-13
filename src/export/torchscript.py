"""TorchScript export backend (``torch.jit.trace``)."""

from __future__ import annotations

import torch

from src.core.entities import ExportRequest
from src.core.ports import ModelExporter
from src.export.registry import exporters


@exporters.register("torchscript")
class TorchScriptExporter(ModelExporter):
    """Trace ``module`` with ``torch.jit.trace`` and save to disk."""

    @property
    def extension(self) -> str:
        return ".pt"

    def export(self, request: ExportRequest) -> None:
        request.path.parent.mkdir(parents=True, exist_ok=True)
        args = request.example_inputs
        with torch.no_grad():
            traced = torch.jit.trace(request.module, args if len(args) > 1 else args[0])
        traced.save(str(request.path))
