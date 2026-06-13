"""ONNX export backend."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from src.core.entities import ExportRequest
from src.core.ports import ModelExporter
from src.export.registry import exporters


def _dynamic_axes(
    input_names: list[str],
    output_names: list[str],
    outputs: tuple[Tensor, ...],
) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {name: {0: "batch"} for name in input_names}
    for name, tensor in zip(output_names, outputs, strict=True):
        entry: dict[int, str] = {0: "batch"}
        if tensor.ndim == 4:
            entry[2] = "height"
            entry[3] = "width"
        axes[name] = entry
    return axes


@exporters.register("onnx")
class OnnxExporter(ModelExporter):
    """Trace ``module`` with ``torch.onnx.export`` (opset configurable)."""

    @property
    def extension(self) -> str:
        return ".onnx"

    def export(self, request: ExportRequest) -> None:
        request.path.parent.mkdir(parents=True, exist_ok=True)
        args = request.example_inputs
        with torch.no_grad():
            raw = request.module(*args) if len(args) > 1 else request.module(args[0])
        outputs = raw if isinstance(raw, tuple) else (raw,)
        dynamic_axes = (
            _dynamic_axes(request.input_names, request.output_names, outputs) if request.dynamic_batch else None
        )
        opset_version = int(request.options.get("opset_version", 17))
        export_args = cast(Any, args if len(args) > 1 else args[0])
        with torch.no_grad():
            torch.onnx.export(
                request.module,
                export_args,
                str(request.path),
                input_names=request.input_names,
                output_names=request.output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                dynamo=False,
            )
