"""TensorRT export: a serialized engine built from the ONNX graph we already write."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import torch

from src.core.choices import one_of
from src.export.backends.onnx import OnnxExporter
from src.export.exporters import Exporter
from src.export.registry import exporter_registry

if TYPE_CHECKING:
    from torch import Tensor

    from src.export.deployable import DeployableModel

log = logging.getLogger(__name__)

type Precision = Literal["fp16", "fp32"]
"""What the engine is allowed to compute in; fp16 trades a looser tolerance for speed."""

_TOLERANCE: dict[str, tuple[float, float]] = {"fp16": (1e-2, 1e-2), "fp32": (1e-4, 1e-3)}
"""Default parity bounds per precision: half precision drifts where a traced graph does not."""

INSTALL_HINT = (
    "TensorRT export needs the 'tensorrt' package, which is deliberately not a declared "
    "dependency: NVIDIA publishes no macOS build, so it cannot be locked from every machine. "
    "Install it on the node that builds the engine: uv pip install tensorrt"
)


def require_tensorrt() -> Any:
    """Return the ``tensorrt`` module, or say exactly how to get it.

    Checked rather than declared, and reported rather than installed: the PyPI
    distribution is a stub that downloads a wheel matching a CUDA version at build
    time, so installing it unattended from inside a run is how an environment ends
    up with an engine its driver cannot load.
    """
    try:
        import tensorrt
    except ImportError as error:
        raise ImportError(INSTALL_HINT) from error
    return tensorrt


@dataclass(frozen=True, slots=True)
class _Engine:
    """Runs a deserialized engine, tensor-in / tensor-out, as the ``Runnable`` contract asks."""

    engine: Any  # tensorrt.ICudaEngine — tensorrt ships no type stubs, so Any is honest
    context: Any  # tensorrt.IExecutionContext

    def __call__(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        trt = require_tensorrt()
        feeds = [tensor.to("cuda").contiguous() for tensor in inputs]
        held: list[Tensor] = []
        written: list[Tensor] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                tensor = feeds[len(held)]
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())
                held.append(tensor)
            else:
                buffer = torch.empty(
                    tuple(self.context.get_tensor_shape(name)),
                    dtype=_torch_dtype(self.engine.get_tensor_dtype(name)),
                    device="cuda",
                )
                self.context.set_tensor_address(name, buffer.data_ptr())
                written.append(buffer)
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return tuple(buffer.detach().cpu() for buffer in written)


@exporter_registry.register("tensorrt")
class TensorRtExporter(Exporter):
    """Builds a serialized TensorRT engine (``.plan``) from the run's deployable graph.

    Through TensorRT's own ONNX parser rather than ``torch_tensorrt``: one dependency, and
    the input is the artifact ``OnnxExporter`` already proves. An engine is hardware- and
    version-specific, so it is built on the node that serves it, and ``tensorrt`` is not a
    declared dependency. **Unverified on macOS**, where NVIDIA ships no build. The
    optimization profile is a batch range: the graph already pins channels and spatial size.

    Parameters:
        precision (str): ``fp16`` or ``fp32``; sets the parity tolerances unless given.
        min_batch (int): Smallest batch the engine accepts.
        opt_batch (int): Batch it is tuned for.
        max_batch (int): Largest batch it accepts.
        opset_version (int): Operator set of the ONNX the engine is parsed from.
        atol (float | None): Absolute parity tolerance; ``None`` follows ``precision``.
        rtol (float | None): Relative parity tolerance; ``None`` follows ``precision``.
        workspace_size (int | None): Builder scratch budget in bytes; ``None`` is TensorRT's default.
    """

    def __init__(
        self,
        precision: Precision = "fp16",
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 8,
        opset_version: int = 18,
        atol: float | None = None,
        rtol: float | None = None,
        workspace_size: int | None = None,
    ) -> None:
        checked = one_of(precision, Precision)
        default_atol, default_rtol = _TOLERANCE[checked]
        super().__init__(atol=default_atol if atol is None else atol, rtol=default_rtol if rtol is None else rtol)
        if not 0 < min_batch <= opt_batch <= max_batch:
            raise ValueError(
                f"The batch profile must satisfy 0 < min <= opt <= max, got {min_batch}/{opt_batch}/{max_batch}."
            )
        require_tensorrt()
        self.precision = checked
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.opset_version = opset_version
        self.workspace_size = workspace_size

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A TensorRT engine is built by the GPU it will run on, and this machine has no CUDA "
                "device. Build it on the node that serves it — the engine is hardware and "
                "TensorRT-version specific, so one built elsewhere would not load anyway."
            )
        trt = require_tensorrt()
        path = destination.parent / f"{destination.name}.plan"
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory() as staging:
            # The intermediate is ours, not the run's: a target that asked for an engine
            # did not ask for an ONNX file beside it.
            graph = OnnxExporter(opset_version=self.opset_version).export(model, example, Path(staging) / "staged")
            path.write_bytes(self._engine_from(trt, graph, example))
        log.info("Built a %s TensorRT engine for batches %d-%d.", self.precision, self.min_batch, self.max_batch)
        return path

    def load(self, path: Path) -> _Engine:
        trt = require_tensorrt()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path.name}; it was built for another GPU or version.")
        return _Engine(engine, engine.create_execution_context())

    def _engine_from(self, trt: Any, graph: Path, example: tuple[Tensor, ...]) -> bytes:
        """Parse the ONNX graph and build the engine, with a profile over the batch axis."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(graph)):
            reasons = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the exported graph: {reasons}")

        config = builder.create_builder_config()
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        if self.workspace_size is not None:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)

        profile = builder.create_optimization_profile()
        for index, tensor in enumerate(example):
            name = network.get_input(index).name
            tail = tuple(tensor.shape[1:])
            profile.set_shape(name, (self.min_batch, *tail), (self.opt_batch, *tail), (self.max_batch, *tail))
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT built no engine from the exported graph; its log above says why.")
        return bytes(engine)


def _torch_dtype(dtype: Any) -> torch.dtype:
    """The torch dtype an engine's tensor declares, for allocating its output buffer."""
    trt = require_tensorrt()
    known: dict[Any, torch.dtype] = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    return known[dtype]
