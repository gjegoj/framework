"""Per-format export option models — the export section's Pydantic contract.

Each export backend has a different parameter surface (ONNX has ``opset_version``
and ``dynamic_batch``; TorchScript does not), so the formats form a Pydantic
*discriminated union* keyed on ``format``. ``extra="forbid"`` on each member makes
a misplaced option (e.g. ``opset_version`` under ``torchscript``) fail at
``load_config`` time — alongside every other parameter — not at export time.

These are pure boundary DTOs: they must NOT import ``src.export`` (Dependency
Rule). The ``format`` discriminator strings mirror the keys in the ``exporters``
registry; a format is added to the union only together with its registered
exporter (so the config can express only what is implemented).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _VerifiableExport(BaseModel):
    """Shared verification knobs for every export backend.

    Per-format because tolerances differ (e.g. fp16 TensorRT needs looser bounds).
    """

    verify_outputs: bool = Field(True, description="Compare exported outputs against the source model.")
    atol: float = Field(1e-4, ge=0.0, description="Max absolute output error tolerated in parity check.")
    rtol: float = Field(1e-3, ge=0.0, description="Max relative output error tolerated in parity check.")

    model_config = ConfigDict(extra="forbid")


class OnnxOptions(_VerifiableExport):
    """ONNX backend options (consumed by ``torch.onnx.export`` and onnxruntime)."""

    format: Literal["onnx"] = "onnx"
    opset_version: int = Field(17, ge=9, description="ONNX opset version.")
    dynamic_batch: bool = Field(True, description="Mark the batch dimension as dynamic.")
    simplify: bool = Field(False, description="Run onnx-simplifier (onnxsim) on the exported graph.")
    check_model: bool = Field(True, description="Run onnx.checker on the exported graph.")
    infer_shapes: bool = Field(False, description="Run strict shape inference (validates type/shape consistency).")


class TorchScriptOptions(_VerifiableExport):
    """TorchScript backend options (``torch.jit.trace`` / ``torch.jit.script``)."""

    format: Literal["torchscript"] = "torchscript"
    method: Literal["trace", "script"] = Field(
        "trace",
        description="trace = run with example inputs (default); script = compile (needs a scriptable model).",
    )


class TrtShapes(BaseModel):
    """Explicit min/opt/max input shapes for a TensorRT optimization profile.

    Each is a full ``[N, C, H, W]`` shape for the image input. Reference the run's input
    size instead of hardcoding it, e.g. ``min: [1, 3, "${image_size.0}", "${image_size.1}"]``.
    ``None`` on the option falls back to a batch-only profile (1/4/8) over the example input's
    own ``C, H, W`` (also derived from ``image_size``).
    """

    min: list[int] = Field(..., description="Lower-bound shape [N, C, H, W].")
    opt: list[int] = Field(..., description="Optimization (typical) shape [N, C, H, W].")
    max: list[int] = Field(..., description="Upper-bound shape [N, C, H, W].")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate(self) -> TrtShapes:
        if not (len(self.min) == len(self.opt) == len(self.max)):
            raise ValueError(f"min/opt/max must share a length, got {len(self.min)}/{len(self.opt)}/{len(self.max)}.")
        for axis, (min_size, opt_size, max_size) in enumerate(zip(self.min, self.opt, self.max, strict=True)):
            if not (min_size <= opt_size <= max_size):
                raise ValueError(
                    f"shapes must satisfy min<=opt<=max per axis; axis {axis} is {min_size}/{opt_size}/{max_size}."
                )
        return self


class TensorRtOptions(_VerifiableExport):
    """TensorRT backend options — exports a serialized engine (``.plan``).

    Compiled directly from the PyTorch graph via torch-tensorrt (no ONNX intermediate).
    fp16 needs a looser ``atol`` than the default — set it in ``configs/export/tensorrt.yaml``.
    """

    format: Literal["tensorrt"] = "tensorrt"
    precision: Literal["fp32", "fp16"] = Field("fp16", description="Compute precision (enabled_precisions).")
    shapes: TrtShapes | None = Field(
        None,
        description="Explicit min/opt/max profile for the image input. None → batch 1/4/8 over the example shape.",
    )
    workspace_size: int | None = Field(None, gt=0, description="TensorRT builder workspace budget in bytes.")
    min_block_size: int = Field(5, ge=1, description="Minimum number of ops in a TensorRT subgraph.")


ExportTarget = Annotated[
    OnnxOptions | TorchScriptOptions | TensorRtOptions,
    Field(discriminator="format"),
]


def _default_targets() -> list[ExportTarget]:
    """Default export targets: a single ONNX target with framework defaults."""
    return [OnnxOptions()]


class ExportConfig(BaseModel):
    """Model export settings (ONNX / TorchScript; future TensorRT)."""

    targets: list[ExportTarget] = Field(
        default_factory=_default_targets,
        description="Per-format export targets. Empty list disables export even when run_export is true.",
    )
    combined: bool = Field(True, description="Export one combined graph: image → all task logits.")
    split_components: bool = Field(
        False,
        description="Also export backbone and each head as separate files.",
    )
    output_dir: str | None = Field(
        None,
        description="Directory for exported artifacts. Defaults to '{save_dir}/export' when unset.",
    )
    generic_io_names: bool = Field(
        True,
        description=(
            "Name exported tensors ``input`` / ``output`` (or ``input_0``, ``output_1``, … when "
            "there are multiple). When false, keep semantic names (task names, stream keys)."
        ),
    )
    input_key: str | None = Field(
        None,
        description="Model input alias for export (defaults to the sole image input from data.inputs).",
    )

    model_config = ConfigDict(extra="forbid")
