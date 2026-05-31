"""Unit tests for composition wiring (no Hydra required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.composition.wiring import build_bindings, build_tasks, build_transforms
from src.data.codecs import FloatCodec, LabelIndexCodec, MultiLabelBinarizeCodec


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "project": "test",
        "epochs": 1,
        "batch_size": 4,
        "image_size": [64, 64],
        "data": {
            "source": "data.csv",
            "image_column": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {"label": {"preset": "classification", "target": "label"}},
    }
    base.update(overrides)
    return base


class TestBuildTransforms:
    def test_all_stages_present(self) -> None:
        config = load_config(_minimal_config())
        transforms = build_transforms(config)
        assert set(transforms) == {Stage.TRAIN, Stage.VAL, Stage.TEST, Stage.PREDICT}

    def test_transforms_produce_correct_shape(self) -> None:
        import torch
        from src.core.entities import Sample
        config = load_config(_minimal_config())
        transform = build_transforms(config)[Stage.TRAIN]
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transform.apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 64, 64])


class TestBuildTasks:
    def test_infers_num_classes_from_runtime(self) -> None:
        config = load_config(_minimal_config())
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        assert len(tasks) == 1
        assert tasks[0].name == "label"
        assert tasks[0].head_spec.out_features == 3

    def test_explicit_num_classes_takes_priority(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["num_classes"] = 5
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})  # would infer 3
        tasks = build_tasks(config, runtime)
        assert tasks[0].head_spec.out_features == 5

    def test_missing_num_classes_raises(self) -> None:
        config = load_config(_minimal_config())
        runtime = RuntimeContext()  # empty — no inferred classes
        with pytest.raises(ValueError, match="num_classes for task 'label'"):
            build_tasks(config, runtime)

    def test_regression_uses_dim(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        runtime = RuntimeContext()  # dim bypasses num_classes lookup
        tasks = build_tasks(config, runtime)
        assert tasks[0].head_spec.out_features == 1


class TestBuildBindings:
    def test_classification_gets_label_index_codec(self) -> None:
        config = load_config(_minimal_config())
        bindings = build_bindings(config)
        assert len(bindings) == 1
        assert isinstance(bindings[0].codec, LabelIndexCodec)
        assert bindings[0].column == "label"

    def test_regression_gets_float_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, FloatCodec)

    def test_multilabel_gets_binarize_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"tags": {"preset": "classification", "target": "tags", "objective": "multilabel"}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, MultiLabelBinarizeCodec)

    def test_custom_separator_via_target_codec_spec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
                "target_codec": {"name": "multilabel_binarize", "separator": "|"},
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, MultiLabelBinarizeCodec)
        assert bindings[0].codec._separator == "|"

    def test_target_codec_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["target_codec"] = "label_index"
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].codec, LabelIndexCodec)

    def test_multitask_order_preserved(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "species": {"preset": "classification", "target": "species"},
            "age": {"preset": "classification", "target": "age"},
        }
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"species": 3, "age": 4})
        tasks = build_tasks(config, runtime)
        assert [t.name for t in tasks] == ["species", "age"]
        assert tasks[0].head_spec.out_features == 3
        assert tasks[1].head_spec.out_features == 4


class TestFullWiringSmoke:
    """Wires all components from a config dict (no Hydra, no Trainer)."""

    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            p = image_dir / f"{i}.jpg"
            cv2.imwrite(str(p), arr)
            rows.append({"image_path": str(p), "label": ["cat", "dog", "cow"][i % 3]})
        csv = tmp_path / "data.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return csv

    def test_end_to_end_wiring(self, csv_path: Path) -> None:
        from src.data import CsvDataSource, DataModule
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        raw = _minimal_config()
        raw["data"]["source"] = str(csv_path)
        config = load_config(raw)

        runtime = RuntimeContext(epochs=config.epochs)
        plain_dm = DataModule(
            source=CsvDataSource([str(csv_path)]),
            bindings=build_bindings(config),
            image_column="image_path",
            transforms=build_transforms(config),
            split=config.data.split,
            runtime=runtime,
            batch_size=4,
            seed=0,
        )
        plain_dm.setup()

        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        lit = LitModule(model=model, tasks=tasks, optimizer_builder=OptimizerBuilder(1e-3))

        import torch
        batch = next(iter(plain_dm.train_dataloader()))
        loss = lit.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
