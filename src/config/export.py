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

from pydantic import BaseModel, ConfigDict, Field


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


ExportTarget = Annotated[
    OnnxOptions | TorchScriptOptions,
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
