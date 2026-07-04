"""Export config schema (src.config.export): per-format discriminated union + YAML files."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_config
from src.config.export import ExportConfig
from tests.support.builders import raw_config as _raw


class TestExportConfig:
    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(export={"targets": [{"format": "coreml"}]}))

    def test_tensorrt_format_accepted(self) -> None:
        config = load_config(_raw(export={"targets": [{"format": "tensorrt", "precision": "fp16"}]}))
        target = config.export.targets[0]
        assert target.format == "tensorrt"
        assert target.precision == "fp16"

    def test_tensorrt_shapes_reject_min_above_max(self) -> None:
        with pytest.raises(ConfigError, match="min<=opt<=max"):
            load_config(
                _raw(
                    export={
                        "targets": [
                            {
                                "format": "tensorrt",
                                "shapes": {"min": [9, 3, 8, 8], "opt": [4, 3, 8, 8], "max": [8, 3, 8, 8]},
                            }
                        ]
                    }
                )
            )

    def test_tensorrt_misplaced_onnx_option_rejected(self) -> None:
        with pytest.raises(ConfigError):
            load_config(_raw(export={"targets": [{"format": "tensorrt", "opset_version": 17}]}))

    def test_empty_targets_allowed(self) -> None:
        config = load_config(_raw(export={"targets": []}))
        assert config.export.targets == []

    def test_default_target_is_onnx(self) -> None:
        config = load_config(_raw())
        assert [t.format for t in config.export.targets] == ["onnx"]

    def test_onnx_and_torchscript_targets(self) -> None:
        config = load_config(_raw(export={"targets": [{"format": "onnx"}, {"format": "torchscript"}]}))
        assert [t.format for t in config.export.targets] == ["onnx", "torchscript"]

    def test_generic_io_names_default(self) -> None:
        config = load_config(_raw())
        assert config.export.generic_io_names is True


class TestExportYamlConfigs:
    def test_all_export_group_files_validate(self) -> None:
        from omegaconf import OmegaConf

        group_dir = Path("configs/export")
        for path in sorted(group_dir.glob("*.yaml")):
            cfg = OmegaConf.load(path)
            if not cfg:  # fully-commented placeholder
                continue
            # Provide image_size so ${image_size.*} interpolations (tensorrt.yaml) resolve.
            resolved = OmegaConf.to_container(OmegaConf.merge({"image_size": [224, 224]}, cfg), resolve=True)
            assert isinstance(resolved, dict)
            resolved.pop("image_size", None)
            ExportConfig.model_validate(resolved)  # must not raise
