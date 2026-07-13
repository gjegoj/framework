"""TorchScript exporter: the trace-time device-portability hazard warning.

Tracing bakes tensors computed in ``forward`` as constants pinned to the trace device.
timm ViTs built with ``dynamic_img_size=True`` compute their rotary position embedding
per-forward, so the traced artifact mixes trace-device constants with ``.to()``-movable
buffers and fails with a device mismatch when loaded on another device. The exporter
detects that model shape and warns with the config fix.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.export.entities import ExportRequest
from src.export.torchscript import TorchScriptExporter


class _DynamicRopeStub(nn.Module):
    """Minimal stand-in for a timm ViT with per-forward (dynamic) rotary embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.dynamic_img_size = True
        self.rope = nn.Identity()  # presence is what the hazard check keys on
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x)
        return out


class _StaticStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.linear(x)
        return out


def _request(module: nn.Module, path: Path) -> ExportRequest:
    return ExportRequest(
        module=module,
        example_inputs=(torch.randn(1, 4),),
        path=path,
        input_names=["input"],
        output_names=["output"],
    )


class TestDynamicRopeHazardWarning:
    def test_dynamic_rope_module_warns_with_config_fix(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        module = nn.Sequential(_DynamicRopeStub())
        with caplog.at_level(logging.WARNING, logger="src.export.torchscript"):
            TorchScriptExporter().export(_request(module, tmp_path / "model.pt"))
        assert any("dynamic_img_size" in record.message for record in caplog.records)
        assert any("map_location" in record.message for record in caplog.records)

    def test_static_module_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="src.export.torchscript"):
            TorchScriptExporter().export(_request(_StaticStub(), tmp_path / "model.pt"))
        assert not [record for record in caplog.records if "dynamic_img_size" in record.message]
