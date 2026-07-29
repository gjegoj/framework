"""Migrated detection task keys: rejected with a pointer to the model section."""

from __future__ import annotations

import pytest

from src.config import ConfigError, load_config
from tests.support.builders import minimal_config


class TestMigratedTaskKeysRejected:
    def test_model_key_rejected_with_pointer(self) -> None:
        raw = minimal_config()
        raw["tasks"]["label"]["model"] = "yolov8n.yaml"
        with pytest.raises(ConfigError, match="moved to the model section"):
            load_config(raw)

    def test_hyperparameters_key_rejected_with_pointer(self) -> None:
        raw = minimal_config()
        raw["tasks"]["label"]["hyperparameters"] = {"box": 3.0}
        with pytest.raises(ConfigError, match="moved to the model section"):
            load_config(raw)
