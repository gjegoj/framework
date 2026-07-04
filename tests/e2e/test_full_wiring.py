"""End-to-end wiring smoke: every component built from one config dict (no Hydra, no Trainer)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from src.composition.wiring import (
    build_bindings,
    build_tasks,
    build_transforms,
)
from src.config import load_config
from src.core.runtime import RuntimeContext
from tests.support.builders import minimal_config as _minimal_config


class TestFullWiringSmoke:
    """Wires all components from a config dict (no Hydra, no Trainer)."""

    @pytest.fixture
    def csv_path(self, make_image_csv: Callable[..., Path]) -> Path:
        return make_image_csv(count=20, size=64, seed=2)

    def test_end_to_end_wiring(self, csv_path: Path) -> None:
        from src.data import CsvDataSource, DataModule
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        raw = _minimal_config()
        raw["data"]["sources"] = str(csv_path)
        config = load_config(raw)

        runtime = RuntimeContext()
        plain_dm = DataModule(
            target_bindings=build_bindings(config),
            inputs_config="image_path",
            transforms=build_transforms(config),
            runtime=runtime,
            batch_size=4,
            seed=0,
            source=CsvDataSource([str(csv_path)]),
            split=config.data.split,
        )
        plain_dm.setup()

        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        lit = LitModule(model=model, tasks=tasks, optimizer_builder=OptimizerBuilder(1e-3))

        batch = next(iter(plain_dm.train_dataloader()))
        result = lit.training_step(batch, 0)
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].ndim == 0
