"""NVIDIA's engine: an ONNX graph compiled for one GPU, one precision and one batch profile."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar, override

import numpy
import torch
from torch import Tensor

from src.export.backends.onnx import OnnxExporter
from src.export.base import ABSOLUTE_TOLERANCE, BATCH_AXIS, RELATIVE_TOLERANCE, Exporter, Runnable
from src.export.deployable import DeployableModel
from src.export.registry import exporter_registry


class Precision(StrEnum):
    """What the engine computes in. Half precision is the reason to build one and the reason it drifts."""

    FULL = "fp32"
    HALF = "fp16"


def require_tensorrt() -> Any:
    """The library, or a refusal naming it — asked once per operation rather than once per tensor.

    It is not a dependency of this framework and cannot become one: it has no build for every platform
    this framework runs on, and it needs an NVIDIA GPU at import. Everything a declaration can be wrong
    about is therefore checked before this is called, so a typo answers on any machine.
    """
    try:
        import tensorrt
    except ImportError as error:
        raise ImportError(
            "tensorrt is not installed, so this run can neither build nor read an engine. It is "
            "`pip install tensorrt` on a machine with an NVIDIA GPU, and it is not a dependency of this "
            "framework because it has no build for every platform the framework runs on."
        ) from error
    return tensorrt


@exporter_registry.register("tensorrt")
class TensorRtExporter(Exporter):
    """An engine built from the ONNX graph its own declaration describes.

    That graph is a *declared* step: the exporter that writes it is a constructor argument, so
    ``{name: tensorrt, onnx: {_target_: ..., opset: 17}}`` builds the engine from a graph at 17. The
    reference implementation constructed one privately and dropped every option a run had declared for
    it, which is how an engine and an artifact of the same run came to disagree. It is written to a
    scratch directory rather than beside the engine, so a run that also declares ``onnx`` keeps the
    artifact its own declaration asked for.

    Nothing below ``require_tensorrt`` has ever run: the library is on no machine this framework is
    developed on, so the engine path is written against the TensorRT 10 API and is unmeasured. What can
    be checked without it — the precision, the batch profile, the ONNX graph it is built from — is
    checked before it is reached. Tolerances are left as declared for the same reason: half precision
    drifts further than the defaults allow, and nobody here can say by how much.

    Parameters:
        precision: What the engine computes in.
        min_batch / opt_batch / max_batch: The batch sizes the engine is built to serve, and the one it
            is optimized for. TensorRT needs all three; the graph it is built from carries a free batch.
        onnx: The ONNX exporter whose graph this engine is compiled from.
    """

    suffix: ClassVar[str] = "engine"

    def __init__(
        self,
        *,
        precision: str = Precision.FULL,
        min_batch: int = 1,
        opt_batch: int = 1,
        max_batch: int = 1,
        onnx: OnnxExporter | None = None,
        atol: float = ABSOLUTE_TOLERANCE,
        rtol: float = RELATIVE_TOLERANCE,
    ) -> None:
        super().__init__(atol=atol, rtol=rtol)
        if precision not in set(Precision):
            raise ValueError(f"Unknown TensorRT precision {precision!r}; an engine is built in {', '.join(Precision)}.")
        if not 0 < min_batch <= opt_batch <= max_batch:
            raise ValueError(
                "A TensorRT engine is built for a batch profile with 0 < min <= opt <= max, and this one "
                f"declares min {min_batch}, opt {opt_batch}, max {max_batch}."
            )
        self.precision = precision
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        if onnx is not None and not isinstance(onnx, OnnxExporter):
            raise TypeError(
                f"`onnx` was declared as {type(onnx).__name__}, and an engine is compiled from an ONNX "
                "graph: TensorRT's parser reads that format and no other. Declare the intermediate step "
                "as an OnnxExporter, or leave it out and this builds its own."
            )
        self.onnx = onnx if onnx is not None else OnnxExporter()

    def describe(self, path: Path) -> Mapping[str, object]:
        """What an engine will and will not do: its precision, and the batch sizes it was built for.

        Both belong in the record because neither can be read off the file without the library that
        built it, and a deployment sending a batch outside the profile gets a refusal, not an answer.
        """
        return {
            "precision": str(self.precision),
            "batch": {"min": self.min_batch, "opt": self.opt_batch, "max": self.max_batch},
        }

    @override
    def answers_at(self, written_at: int) -> tuple[int, ...]:
        """An engine answers inside the profile it was built for and nowhere else, whatever wrote it."""
        return tuple(sorted({self.min_batch, self.opt_batch, self.max_batch}))

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        trt = require_tensorrt()
        with TemporaryDirectory() as scratch:
            source = self.onnx.export(graph, example, Path(scratch) / "graph")
            path.write_bytes(self._compiled(trt, source, example))

    def _compiled(self, trt: Any, source: Path, example: tuple[Tensor, ...]) -> bytes:
        """The serialized engine, with one optimization profile covering the declared batch range."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(source.read_bytes()):
            said = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise ValueError(f"TensorRT could not read the ONNX graph this export wrote: {said}")
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        for index, tensor in enumerate(example):
            rest = tuple(tensor.shape)[BATCH_AXIS + 1 :]
            name = network.get_input(index).name
            profile.set_shape(name, (self.min_batch, *rest), (self.opt_batch, *rest), (self.max_batch, *rest))
        config.add_optimization_profile(profile)
        if self.precision == Precision.HALF:
            config.set_flag(trt.BuilderFlag.FP16)
        built = builder.build_serialized_network(network, config)
        if built is None:
            raise RuntimeError(
                f"TensorRT built no engine from {source.name}; its log above says why — an unsupported "
                "operator and a profile this GPU has no memory for are the two usual reasons."
            )
        return bytes(built)

    def load(self, path: Path) -> Runnable:
        trt = require_tensorrt()
        engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(path.read_bytes())
        context = engine.create_execution_context()
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        fed = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        answers = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

        def run(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            given = [one.detach().to("cuda").contiguous() for one in tensors]
            for name, tensor in zip(fed, given, strict=True):
                context.set_input_shape(name, tuple(tensor.shape))
                context.set_tensor_address(name, tensor.data_ptr())
            written = []
            for name in answers:
                dtype = torch.from_numpy(numpy.empty(0, dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
                buffer = torch.empty(tuple(context.get_tensor_shape(name)), dtype=dtype, device="cuda")
                context.set_tensor_address(name, buffer.data_ptr())
                written.append(buffer)
            stream = torch.cuda.current_stream()
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            return tuple(one.cpu() for one in written)

        return run
