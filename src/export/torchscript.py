"""TorchScript export backend (``torch.jit.trace`` / ``torch.jit.script``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from src.core.entities import ExportRequest
from src.core.ports import ModelExporter
from src.export.registry import exporters


@dataclass
class _TorchScriptRunnable:
    """Adapter: run a written TorchScript file, tensor-in / tensor-out."""

    module: torch.jit.ScriptModule

    def __call__(self, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            raw = self.module(*inputs.values())
        return raw if isinstance(raw, tuple) else (raw,)


@exporters.register("torchscript")
class TorchScriptExporter(ModelExporter):
    """Compile ``module`` via ``torch.jit`` (``method`` = trace or script) and save to disk."""

    @property
    def extension(self) -> str:
        return ".pt"

    def export(self, request: ExportRequest) -> None:
        request.path.parent.mkdir(parents=True, exist_ok=True)
        if request.options.get("method", "trace") == "script":
            compiled = self._script(request.module)
        else:
            args = request.example_inputs
            with torch.no_grad():
                compiled = torch.jit.trace(request.module, args if len(args) > 1 else args[0])
        compiled.save(str(request.path))

    @staticmethod
    def _script(module: torch.nn.Module) -> torch.jit.ScriptModule:
        """Compile via ``torch.jit.script``, surfacing a readable error on failure."""
        try:
            return torch.jit.script(module)
        except Exception as error:  # noqa: BLE001 — torch raises varied, cryptic errors here
            raise RuntimeError(
                "TorchScript method='script' could not compile the model. The composite export "
                "wrappers use dataclasses (FeatureBundle / ModelOutput) the TorchScript compiler "
                "cannot handle — use method='trace' (the default) for these models."
            ) from error

    def load(self, path: Path) -> _TorchScriptRunnable:
        module = torch.jit.load(str(path))
        module.eval()
        return _TorchScriptRunnable(module)
