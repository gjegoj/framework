"""TorchScript export backend (``torch.jit.trace`` / ``torch.jit.script``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from src.export.entities import ExportRequest
from src.export.ports import ModelExporter
from src.export.registry import exporters
from src.export.tracing import as_output_tuple, trace_args

log = logging.getLogger(__name__)


def _warn_on_dynamic_rope(module: torch.nn.Module) -> None:
    """Warn when tracing a ViT whose rotary position embedding is computed per-forward.

    ``torch.jit.trace`` bakes tensors computed inside ``forward`` as constants pinned to the
    trace device. timm ViTs built with ``dynamic_img_size=True`` (the default for the
    DINOv3/EVA family) recompute their ROPE coordinate grid every forward, so the traced
    artifact mixes trace-device constants with ``.to()``-movable buffers and fails with
    "Expected all tensors to be on the same device" when moved to another device. Built
    statically (``dynamic_img_size=False`` + a fixed ``img_size``), the embedding lives in a
    registered buffer (``pos_embed_cached``) that follows ``.to()``.
    """
    for submodule in module.modules():
        if getattr(submodule, "dynamic_img_size", False) and getattr(submodule, "rope", None) is not None:
            log.warning(
                "Tracing %s with dynamic_img_size=True and rotary position embeddings: the ROPE grid "
                "is computed per-forward and will be baked as constants on the trace device, so the "
                "exported TorchScript is NOT device-portable via .to(device) (consumers must load it "
                "with torch.jit.load(path, map_location=device)). For a portable artifact set "
                "`dynamic_img_size: false` and `img_size: ${image_size}` in the backbone config so "
                "the embedding is cached in a buffer.",
                type(submodule).__name__,
            )
            return


@dataclass
class _TorchScriptRunnable:
    """Adapter: run a written TorchScript file, tensor-in / tensor-out."""

    module: torch.jit.ScriptModule

    def __call__(self, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
        with torch.no_grad():
            raw = self.module(*inputs.values())
        return as_output_tuple(raw)


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
            _warn_on_dynamic_rope(request.module)
            with torch.no_grad():
                compiled = torch.jit.trace(request.module, trace_args(request.example_inputs))
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
