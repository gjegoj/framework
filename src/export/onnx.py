"""ONNX export backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from src.export.entities import ExportRequest
from src.export.ports import ModelExporter
from src.export.registry import exporters
from src.export.tracing import as_output_tuple, trace_args


@dataclass
class _OnnxRunnable:
    """Adapter: run a written ONNX file via onnxruntime, tensor-in / tensor-out."""

    session: Any  # onnxruntime.InferenceSession — onnxruntime ships no type stubs, so Any is honest

    def __call__(self, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
        feeds = {name: tensor.detach().cpu().numpy() for name, tensor in inputs.items()}
        outputs = self.session.run(None, feeds)
        return tuple(torch.from_numpy(output) for output in outputs)


def _simplify_onnx(path: Path) -> None:
    """Run onnx-simplifier on an exported file, rewriting it in place.

    Lazy-imports onnx/onnxsim so the dependency is only touched when a target
    opts in. Raises if onnxsim cannot validate the simplified graph: a model the
    user asked to simplify but that failed should fail loudly, not ship
    un-simplified silently.

    Parameters:
        path (Path): The exported ``.onnx`` file to simplify in place.
    """
    import onnx
    from onnxsim import simplify

    simplified, ok = simplify(onnx.load(str(path)))
    if not ok:
        raise RuntimeError(f"onnxsim could not validate the simplified model for {path}.")
    onnx.save(simplified, str(path))


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
        dynamic_batch = bool(request.options.get("dynamic_batch", False))
        dynamic_axes = None
        if dynamic_batch:  # the dry-run forward only feeds dynamic-axis inference; skip it otherwise
            with torch.no_grad():
                outputs = as_output_tuple(request.module(*args))
            dynamic_axes = _dynamic_axes(request.input_names, request.output_names, outputs)
        opset_version = int(request.options.get("opset_version", 17))
        with torch.no_grad():
            torch.onnx.export(
                request.module,
                cast(Any, trace_args(args)),
                str(request.path),
                input_names=request.input_names,
                output_names=request.output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                dynamo=False,
            )
        if bool(request.options.get("simplify", False)):
            _simplify_onnx(request.path)

    def load(self, path: Path) -> _OnnxRunnable:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _OnnxRunnable(session)

    def validate(self, request: ExportRequest) -> dict[str, str]:
        import onnx
        from onnx import shape_inference

        checks: dict[str, str] = {}
        model = onnx.load(str(request.path))
        if bool(request.options.get("check_model", True)):
            try:
                onnx.checker.check_model(model)
                checks["onnx.checker"] = ""
            except Exception as error:  # noqa: BLE001 — record any checker failure as the detail
                checks["onnx.checker"] = str(error)
        if bool(request.options.get("infer_shapes", False)):
            try:
                shape_inference.infer_shapes(model, strict_mode=True)
                checks["shape_inference"] = ""
            except Exception as error:  # noqa: BLE001
                checks["shape_inference"] = str(error)
        return checks
