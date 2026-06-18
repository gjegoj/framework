"""TensorRT export backend — a serialized engine (``.plan``) for Triton's ``tensorrt_plan``.

Compiled straight from the PyTorch graph via torch-tensorrt (no ONNX intermediate), so it
slots into the same ``ModelExporter`` seam as ONNX/TorchScript. CUDA-only: the engine is built
and run on the GPU, and it is hardware + TensorRT-version specific — build it on a node matching
the deployment. ``torch_tensorrt`` / ``tensorrt`` are optional (lazy-imported).

Targets torch-tensorrt 2.x / TensorRT 10.x — the engine-build / runtime calls may need a small
tweak on other versions; everything else (device handling, profile shapes) is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.core.entities import ExportRequest
from src.core.ports import ModelExporter
from src.export.registry import exporters

# Default batch profile when no explicit ``shapes`` is given: min / opt / max batch over the
# example input's own C, H, W (themselves derived from config image_size / mean).
_DEFAULT_BATCH = (1, 4, 8)


def _profile_shapes(options: dict[str, Any], example: Tensor) -> tuple[list[int], list[int], list[int]]:
    """Resolve the (min, opt, max) input shapes for the optimization profile."""
    shapes = options.get("shapes")
    if shapes is not None:
        return list(shapes["min"]), list(shapes["opt"]), list(shapes["max"])
    chw = list(example.shape[1:])
    return [_DEFAULT_BATCH[0], *chw], [_DEFAULT_BATCH[1], *chw], [_DEFAULT_BATCH[2], *chw]


def _trt_to_torch_dtype(trt_dtype: Any) -> torch.dtype:
    """Map a ``tensorrt.DataType`` to its torch dtype (for allocating output buffers)."""
    import tensorrt as trt

    mapping: dict[Any, torch.dtype] = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = torch.int64
    return mapping[trt_dtype]


@dataclass
class _TensorRtRunnable:
    """Run a deserialized TensorRT engine, tensor-in / tensor-out (for parity checks)."""

    engine: Any  # tensorrt.ICudaEngine — tensorrt ships no type stubs, so Any is honest
    context: Any  # tensorrt.IExecutionContext

    def __call__(self, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
        import tensorrt as trt

        feeds = [tensor.to("cuda").contiguous() for tensor in inputs.values()]
        held: list[Tensor] = []
        outputs: list[Tensor] = []
        next_input = 0
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                tensor = feeds[next_input]
                next_input += 1
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())
                held.append(tensor)
            else:
                shape = tuple(self.context.get_tensor_shape(name))
                buffer = torch.empty(
                    shape, dtype=_trt_to_torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda"
                )
                self.context.set_tensor_address(name, buffer.data_ptr())
                outputs.append(buffer)
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return tuple(buffer.detach().cpu() for buffer in outputs)


@exporters.register("tensorrt")
class TensorRtExporter(ModelExporter):
    """Compile ``module`` to a serialized TensorRT engine (``.plan``)."""

    @property
    def extension(self) -> str:
        return ".plan"

    def export(self, request: ExportRequest) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT export requires a CUDA device. Run export on the GPU / TensorRT node that "
                "matches your Triton deployment (the engine is hardware + TRT-version specific)."
            )
        import torch_tensorrt

        request.path.parent.mkdir(parents=True, exist_ok=True)
        options = request.options
        min_shape, opt_shape, max_shape = _profile_shapes(options, request.example_inputs[0])
        precision = str(options.get("precision", "fp16"))
        enabled = {torch.float16} if precision == "fp16" else {torch.float32}

        module = request.module.to("cuda").eval()
        try:
            trt_input = torch_tensorrt.Input(
                min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape, dtype=torch.float32
            )
            traced = torch.jit.trace(module, torch.randn(*opt_shape, device="cuda"))
            kwargs: dict[str, Any] = {
                "inputs": [trt_input],
                "enabled_precisions": enabled,
                "min_block_size": int(options.get("min_block_size", 5)),
            }
            workspace = options.get("workspace_size")
            if workspace:
                kwargs["workspace_size"] = int(workspace)
            engine = torch_tensorrt.convert_method_to_trt_engine(traced, "forward", **kwargs)
            request.path.write_bytes(engine)
        finally:
            module.to("cpu")  # restore the pipeline's CPU module (shared across targets)

    def load(self, path: Path) -> _TensorRtRunnable:
        import tensorrt as trt

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        return _TensorRtRunnable(engine, engine.create_execution_context())
