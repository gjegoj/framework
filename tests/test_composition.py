"""Unit tests for composition wiring (no Hydra required)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.callbacks.batch_transform import BatchTransformCallback
from src.composition.wiring import (
    build_bindings,
    build_data_source,
    build_lit_module,
    build_logger,
    build_task_lr_overrides,
    build_tasks,
    build_transforms,
)
from src.config import load_config
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.encoders import LabelEncoder, MultiLabelEncoder, ScalarEncoder
from src.data.sources import CsvDataSource, JsonDataSource

_RESIZE_NORMALIZE_TOTENSOR = [
    {"_target_": "albumentations.Resize", "height": 64, "width": 64},
    {"_target_": "albumentations.Normalize"},
    {"_target_": "albumentations.pytorch.ToTensorV2"},
]

_MINIMAL_TRANSFORMS: dict[str, Any] = {
    "train": {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR},
    "val": {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR},
}


def _minimal_config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "project": "test",
        "epochs": 1,
        "batch_size": 4,
        "lr": 1e-3,
        "image_size": [64, 64],
        "data": {
            "sources": "data.csv",
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "backbone": {"name": "resnet18"},
        "optimizer": {"lr": 1e-3},
        "tasks": {
            "label": {"preset": "classification", "target": "label", "class_mapping": {0: "cat", 1: "cow", 2: "dog"}}
        },
        "transforms": _MINIMAL_TRANSFORMS,
    }
    base.update(overrides)
    return base


class TestSourceConfigGrammar:
    def test_plain_string_source_unchanged(self) -> None:
        config = load_config(_minimal_config())  # sources: "data.csv"
        assert config.data.sources == "data.csv"

    def test_list_mixes_plain_paths_and_per_source_objects(self) -> None:
        from src.config.schema import SourceConfig

        raw = _minimal_config()
        raw["data"] = {
            "sources": [
                "data/a.csv",
                {"path": "data/b.csv", "transforms": {"train": {"_target_": "albumentations.HorizontalFlip"}}},
            ],
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
        sources = load_config(raw).data.sources
        assert isinstance(sources, list)
        assert sources[0] == "data/a.csv"
        assert isinstance(sources[1], SourceConfig)
        assert sources[1].path == "data/b.csv"
        assert sources[1].transforms == {"train": {"_target_": "albumentations.HorizontalFlip"}}

    def test_source_config_forbids_unknown_keys(self) -> None:
        from pydantic import ValidationError

        from src.config.schema import SourceConfig

        with pytest.raises(ValidationError):
            SourceConfig(path="a.csv", bogus=1)  # type: ignore[call-arg]

    def test_presplit_dict_still_accepted(self) -> None:
        raw = _minimal_config()
        raw["data"] = {"sources": {"train": "tr.csv", "val": "va.csv"}, "inputs": "image_path"}
        config = load_config(raw)
        assert config.data.sources == {"train": "tr.csv", "val": "va.csv"}

    def test_split_source_rejects_single_transform(self) -> None:
        from src.config import ConfigError

        raw = _minimal_config()
        raw["data"] = {
            "sources": ["a.csv", {"path": "b.csv", "transforms": {"_target_": "albumentations.HorizontalFlip"}}],
            "inputs": "image_path",
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        }
        with pytest.raises(ConfigError, match="per-stage dict"):
            load_config(raw)  # split source spans stages → needs per-stage dict, not a single transform

    def test_presplit_source_rejects_per_stage_transform(self) -> None:
        from src.config import ConfigError

        raw = _minimal_config()
        raw["data"] = {
            "sources": {
                "train": [{"path": "a.csv", "transforms": {"train": {"_target_": "albumentations.HorizontalFlip"}}}],
                "val": "v.csv",
            },
            "inputs": "image_path",
        }
        with pytest.raises(ConfigError, match="single transform"):
            load_config(raw)  # pre-split source is pinned to its stage → needs a single transform

    def test_presplit_source_accepts_single_transform(self) -> None:
        from src.config.schema import SourceConfig

        raw = _minimal_config()
        raw["data"] = {
            "sources": {
                "train": ["a.csv", {"path": "b.csv", "transforms": {"_target_": "albumentations.HorizontalFlip"}}],
                "val": "v.csv",
            },
            "inputs": "image_path",
        }
        sources = load_config(raw).data.sources
        assert isinstance(sources, dict)
        train_value = sources["train"]
        assert isinstance(train_value, list)
        train_entry = train_value[1]
        assert isinstance(train_entry, SourceConfig)
        assert train_entry.is_single_transform is True

    def test_malformed_transforms_rejected(self) -> None:
        from pydantic import ValidationError

        from src.config.schema import SourceConfig

        with pytest.raises(ValidationError):
            SourceConfig(path="a.csv", transforms={"trai": {"_target_": "x"}})  # typo'd stage key, no _target_ at top


class TestBuildSourceBindings:
    def _config(self, sources: Any) -> Any:
        raw = _minimal_config()
        raw["data"] = {"sources": sources, "inputs": "image_path", "split": {"train": 0.8, "val": 0.1, "test": 0.1}}
        return load_config(raw)

    def test_plain_list_collapses_to_one_binding(self) -> None:
        from src.composition.wiring import build_source_bindings

        config = self._config(["a.csv", "b.csv"])
        global_transforms = build_transforms(config, [])
        bindings = build_source_bindings(config, global_transforms, [])
        assert len(bindings) == 1  # one source over all paths → current behaviour (global split)
        assert bindings[0].transforms[Stage.TRAIN] is global_transforms[Stage.TRAIN]

    def test_per_source_override_resolves_with_global_fallback(self) -> None:
        from src.composition.wiring import build_source_bindings

        override = {"train": {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR}}
        config = self._config(["a.csv", {"path": "b.csv", "transforms": override}])
        global_transforms = build_transforms(config, [])
        bindings = build_source_bindings(config, global_transforms, [])
        assert len(bindings) == 2
        assert bindings[0].transforms[Stage.TRAIN] is global_transforms[Stage.TRAIN]  # plain source: all global
        # overridden source: train is its own transform, val/test fall back to global
        assert bindings[1].transforms[Stage.TRAIN] is not global_transforms[Stage.TRAIN]
        assert bindings[1].transforms[Stage.VAL] is global_transforms[Stage.VAL]


class TestBuildStagedSources:
    def test_returns_none_in_split_mode(self) -> None:
        from src.composition.wiring import build_staged_sources

        config = load_config(_minimal_config())  # split mode
        assert build_staged_sources(config, build_transforms(config, []), []) is None

    def test_per_stage_bindings_with_single_override_and_fallback(self) -> None:
        from src.composition.wiring import build_staged_sources

        single = {"_target_": "albumentations.Compose", "transforms": _RESIZE_NORMALIZE_TOTENSOR}
        raw = _minimal_config()
        raw["data"] = {
            "sources": {"train": ["a.csv", {"path": "b.csv", "transforms": single}], "val": "v.csv"},
            "inputs": "image_path",
        }
        config = load_config(raw)
        global_transforms = build_transforms(config, [])
        staged = build_staged_sources(config, global_transforms, [])
        assert staged is not None
        train = staged[Stage.TRAIN]
        assert len(train) == 2
        assert train[0].transforms[Stage.TRAIN] is global_transforms[Stage.TRAIN]  # source a → global
        assert train[1].transforms[Stage.TRAIN] is not global_transforms[Stage.TRAIN]  # source b → its override
        assert staged[Stage.VAL][0].transforms[Stage.VAL] is global_transforms[Stage.VAL]  # val → global


class TestInstantiateNested:
    def test_builds_albumentations_resize(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {"_target_": "albumentations.Resize", "height": 32, "width": 32}
        obj = instantiate(spec)
        assert isinstance(obj, A.Resize)

    def test_builds_compose_with_nested_transforms(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {
            "_target_": "albumentations.Compose",
            "transforms": [
                {"_target_": "albumentations.HorizontalFlip", "p": 0.5},
                {"_target_": "albumentations.Normalize"},
            ],
        }
        obj = instantiate(spec)
        assert isinstance(obj, A.Compose)
        assert len(obj.transforms) == 2

    def test_drops_hydra_meta_keys(self) -> None:
        import albumentations as A

        from src.core.instantiate import instantiate

        spec = {
            "_target_": "albumentations.HorizontalFlip",
            "_convert_": "partial",
            "p": 0.3,
        }
        obj = instantiate(spec)
        assert isinstance(obj, A.HorizontalFlip)

    def test_plain_values_pass_through(self) -> None:
        from src.core.instantiate import instantiate

        assert instantiate(42) == 42
        assert instantiate("hello") == "hello"
        assert instantiate([1, 2, 3]) == [1, 2, 3]


class TestResolveSpecClass:
    def test_string_resolves_registry_class(self) -> None:
        from src.core.instantiate import resolve_spec_class
        from src.losses.criterion import CrossEntropyCriterion
        from src.losses.registry import criteria

        assert resolve_spec_class("cross_entropy", criteria) is CrossEntropyCriterion

    def test_name_mapping_resolves_registry_class(self) -> None:
        from src.core.instantiate import resolve_spec_class
        from src.losses.registry import criteria

        resolved = resolve_spec_class({"name": "cross_entropy", "label_smoothing": 0.1}, criteria)
        assert resolved.__name__ == "CrossEntropyCriterion"

    def test_target_resolves_imported_class(self) -> None:
        from src.core.instantiate import resolve_spec_class

        assert resolve_spec_class({"_target_": "torch.optim.SGD"}).__name__ == "SGD"

    def test_mapping_without_name_or_target_raises(self) -> None:
        from src.core.instantiate import resolve_spec_class

        with pytest.raises(ValueError, match="name.*_target_|_target_.*name"):
            resolve_spec_class({"margin": 0.5})


class TestBuildTransforms:
    def test_all_stages_present(self) -> None:
        config = load_config(_minimal_config())
        transforms = build_transforms(config)
        assert set(transforms) == {Stage.TRAIN, Stage.VAL, Stage.TEST, Stage.PREDICT}

    def test_non_albumentations_transform_used_directly(self) -> None:
        from src.transforms.sample import IdentityTransform

        raw = _minimal_config()
        raw["transforms"] = {
            "train": {"_target_": "src.transforms.sample.IdentityTransform"},
            "val": {"_target_": "src.transforms.sample.IdentityTransform"},
        }
        config = load_config(raw)
        transforms = build_transforms(config)
        assert isinstance(transforms[Stage.TRAIN], IdentityTransform)

    def test_transforms_produce_correct_shape(self) -> None:
        import torch

        from src.core.entities import Sample

        config = load_config(_minimal_config())
        transform = build_transforms(config)[Stage.TRAIN]
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transform.apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 64, 64])

    def test_config_transforms_used_when_present(self) -> None:
        import torch

        from src.core.entities import Sample

        raw = _minimal_config()
        raw["transforms"] = {
            "train": {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {"_target_": "albumentations.Resize", "height": 16, "width": 16},
                    {"_target_": "albumentations.Normalize"},
                    {"_target_": "albumentations.pytorch.ToTensorV2"},
                ],
            },
            "val": {
                "_target_": "albumentations.Compose",
                "transforms": [
                    {"_target_": "albumentations.Resize", "height": 16, "width": 16},
                    {"_target_": "albumentations.Normalize"},
                    {"_target_": "albumentations.pytorch.ToTensorV2"},
                ],
            },
        }
        config = load_config(raw)
        transforms = build_transforms(config)
        assert set(transforms) >= {Stage.TRAIN, Stage.VAL}
        # test and predict derived from val
        assert Stage.TEST in transforms
        assert Stage.PREDICT in transforms

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transforms[Stage.TRAIN].apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 16, 16])


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
    def test_classification_gets_label_encoder(self) -> None:
        config = load_config(_minimal_config())
        bindings = build_bindings(config)
        assert len(bindings) == 1
        assert isinstance(bindings[0].encoder, LabelEncoder)
        assert bindings[0].column == "label"

    def test_regression_gets_float_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {"score": {"preset": "regression", "target": "score", "dim": 1}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, ScalarEncoder)

    def test_targetless_task_gets_null_encoder_and_no_column(self) -> None:
        from src.data.encoders import NullTargetEncoder

        raw = _minimal_config()
        raw["tasks"] = {"rank": {"preset": "triplet"}}  # no 'target' → structure-only supervision
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, NullTargetEncoder)
        assert bindings[0].column is None

    def test_multilabel_gets_binarize_codec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MultiLabelEncoder)

    def test_custom_separator_via_target_encoder_spec(self) -> None:
        raw = _minimal_config()
        raw["tasks"] = {
            "tags": {
                "preset": "classification",
                "target": "tags",
                "objective": "multilabel",
                "target_encoder": {"name": "multilabel", "separator": "|"},
            }
        }
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MultiLabelEncoder)
        assert bindings[0].encoder._separator == "|"

    def test_target_encoder_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["target_encoder"] = "label"
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, LabelEncoder)

    def test_segmentation_gets_mask_codec(self) -> None:
        from src.data.encoders import MaskEncoder

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        bindings = build_bindings(config)
        assert isinstance(bindings[0].encoder, MaskEncoder)


class TestDataConfigModes:
    def test_split_mode_str(self) -> None:
        config = load_config(_minimal_config())
        assert config.data.sources == "data.csv"
        assert config.data.split is not None

    def test_split_mode_list(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": ["a.csv", "b.csv"],
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(config.data.sources, list)

    def test_presplit_mode_dict(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": {"train": "train.csv", "val": "val.csv"},
                    "inputs": "image_path",
                }
            )
        )
        assert isinstance(config.data.sources, dict)
        assert config.data.split is None

    def test_presplit_list_of_paths_per_stage(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": {"train": ["a.csv", "b.csv"], "val": "val.csv"},
                    "inputs": "image_path",
                }
            )
        )
        assert isinstance(config.data.sources, dict)
        assert isinstance(config.data.sources["train"], list)

    def test_presplit_with_split_raises(self) -> None:
        with pytest.raises(Exception, match="split"):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"train": "train.csv", "val": "val.csv"},
                        "inputs": "image_path",
                        "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                    }
                )
            )

    def test_split_mode_without_split_raises(self) -> None:
        with pytest.raises(Exception, match="split"):
            load_config(
                _minimal_config(
                    data={
                        "sources": "data.csv",
                        "inputs": "image_path",
                    }
                )
            )

    def test_presplit_missing_train_raises(self) -> None:
        with pytest.raises(Exception, match="train"):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"val": "val.csv"},
                        "inputs": "image_path",
                    }
                )
            )

    def test_invalid_stage_key_raises(self) -> None:
        with pytest.raises(Exception):
            load_config(
                _minimal_config(
                    data={
                        "sources": {"train": "train.csv", "predict": "pred.csv"},
                        "inputs": "image_path",
                    }
                )
            )


class TestBuildDataSource:
    def test_infers_csv_from_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.csv",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), CsvDataSource)

    def test_infers_json_from_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.json",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), JsonDataSource)

    def test_explicit_source_type_overrides_extension(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.dat",
                    "source_type": "csv",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        assert isinstance(build_data_source(config.data), CsvDataSource)

    def test_unknown_extension_raises(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": "data/x.parquet",
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown source extension"):
            build_data_source(config.data)

    def test_mixed_extensions_raise(self) -> None:
        config = load_config(
            _minimal_config(
                data={
                    "sources": ["a.csv", "b.json"],
                    "inputs": "image_path",
                    "split": {"train": 0.8, "val": 0.1, "test": 0.1},
                }
            )
        )
        with pytest.raises(ValueError, match="mixed extensions"):
            build_data_source(config.data)

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


class TestBuildTaskLrOverrides:
    def test_no_overrides_returns_empty(self) -> None:
        config = load_config(_minimal_config())
        assert build_task_lr_overrides(config) == {}

    def test_single_task_override(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        overrides = build_task_lr_overrides(config)
        assert overrides == {"label": pytest.approx(1e-4)}

    def test_partial_override_only_includes_declared_tasks(self) -> None:
        raw = _minimal_config()
        raw["tasks"]["species"] = {"preset": "classification", "target": "species", "optimizer": {"lr": 5e-5}}
        raw["tasks"]["age"] = {"preset": "classification", "target": "age"}
        config = load_config(raw)
        overrides = build_task_lr_overrides(config)
        assert set(overrides.keys()) == {"species"}
        assert overrides["species"] == pytest.approx(5e-5)


class TestBuildLitModule:
    def test_creates_lit_module_and_serialises_hparams(self) -> None:
        # Per-task LR lives on the OptimizerBuilder now (see TestOptimizerBuilder); build_lit_module
        # only wires collaborators + hparams, so it no longer branches on per-task overrides.
        from src.models import build_composite_model
        from src.models.registry import backbones
        from src.training import LitModule, OptimizerBuilder

        config = load_config(_minimal_config())
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})

        lit = build_lit_module(config, model, tasks, OptimizerBuilder(base_lr=1e-3))
        assert isinstance(lit, LitModule)
        assert lit._hparams_to_log is not None  # full config serialised as hyperparams

    def test_param_groups_reflect_per_task_lr(self) -> None:
        from src.composition.wiring import build_optimizer_builder
        from src.models import build_composite_model
        from src.models.registry import backbones

        raw = _minimal_config()
        raw["tasks"]["label"]["optimizer"] = {"lr": 1e-4}
        config = load_config(raw)
        runtime = RuntimeContext(num_classes={"label": 3})
        tasks = build_tasks(config, runtime)
        backbone = backbones.create("timm", name="resnet18", pretrained=False)
        model = build_composite_model(backbone, {t.name: t.head_spec for t in tasks})
        opt_builder = build_optimizer_builder(config.optimizer, build_task_lr_overrides(config))

        lit = build_lit_module(config, model, tasks, opt_builder)
        result = lit.configure_optimizers()
        assert isinstance(result, torch.optim.Optimizer)
        lrs = {g["name"]: g["lr"] for g in result.param_groups}
        assert lrs["backbone"] == pytest.approx(1e-3)
        assert lrs["head/label"] == pytest.approx(1e-4)


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

        import torch

        batch = next(iter(plain_dm.train_dataloader()))
        result = lit.training_step(batch, 0)
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].ndim == 0


class TestBuildLogger:
    def test_none_kind_returns_false(self) -> None:
        config = load_config(_minimal_config())
        assert config.logger.kind == "none"
        assert build_logger(config) is False

    def test_default_logger_config_is_none(self) -> None:
        from src.config.schema import LoggerConfig

        cfg = LoggerConfig()
        assert cfg.kind == "none"
        assert cfg.project is None
        assert cfg.task is None

    def test_unknown_kind_raises(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "wandb"}
        config = load_config(raw)
        with pytest.raises(ValueError, match="Unknown logger kind"):
            build_logger(config)

    def test_logger_config_parses_clearml_kind(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "project": "ml-tests"}
        config = load_config(raw)
        assert config.logger.kind == "clearml"
        assert config.logger.project == "ml-tests"

    def test_clearml_logger_is_both_instances(self) -> None:
        """ClearMLLogger must satisfy Lightning's Logger and every artifact-logger port."""
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        import lightning as L

        from src.core.ports import (
            CurveLogger,
            HistogramLogger,
            HtmlLogger,
            MatrixLogger,
            PlotLogger,
            SingleValueLogger,
        )
        from src.loggers.clearml import ClearMLLogger

        mock_task = MagicMock()
        mock_task.name = "test-task"
        mock_task.id = "abc-123"
        mock_task.get_logger.return_value = MagicMock()

        with patch("clearml.Task.init", return_value=mock_task):
            logger = ClearMLLogger(project_name="test-proj", task_name="test-task")

        assert isinstance(logger, L.pytorch.loggers.Logger)
        for port in (MatrixLogger, CurveLogger, HtmlLogger, SingleValueLogger, HistogramLogger, PlotLogger):
            assert isinstance(logger, port)

    def test_clearml_log_html_reports_media(self) -> None:
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        from src.loggers.clearml import ClearMLLogger

        mock_task = MagicMock()
        mock_backend = MagicMock()
        mock_task.name = "t"
        mock_task.id = "i"
        mock_task.get_logger.return_value = mock_backend
        with patch("clearml.Task.init", return_value=mock_task):
            logger = ClearMLLogger(project_name="p", task_name="t")

        logger.log_html("samples/val", "<html><body>hello</body></html>", iteration=2)
        mock_backend.report_media.assert_called_once()
        kwargs = mock_backend.report_media.call_args.kwargs
        assert kwargs["title"] == "samples/val"
        assert kwargs["iteration"] == 2
        assert kwargs["file_extension"] == "html"
        assert "hello" in kwargs["stream"].getvalue()

    def test_clearml_split_metric_name_multipart(self) -> None:
        pytest.importorskip("clearml")
        from src.loggers.clearml import ClearMLLogger

        title, series = ClearMLLogger._split_metric_name("label/f1/val")
        assert title == "label/f1"
        assert series == "val"

    def test_clearml_split_metric_name_single(self) -> None:
        pytest.importorskip("clearml")
        from src.loggers.clearml import ClearMLLogger

        title, series = ClearMLLogger._split_metric_name("loss")
        assert title == "loss"
        assert series == "value"

    def test_logger_config_parses_tags(self) -> None:
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", "resnet18", "lr=0.001"]}
        config = load_config(raw)
        assert config.logger.tags == ["timm", "resnet18", "lr=0.001"]

    def test_logger_config_drops_empty_tags(self) -> None:
        """A None tag (e.g. ${backbone.name} on a multi backbone) or empty string is dropped."""
        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", None, "", "lr=0.001"]}
        config = load_config(raw)
        assert config.logger.tags == ["timm", "lr=0.001"]

    def test_clearml_builder_forwards_tags(self) -> None:
        pytest.importorskip("clearml")
        from unittest.mock import MagicMock, patch

        raw = _minimal_config()
        raw["logger"] = {"kind": "clearml", "tags": ["timm", "lr=0.001"]}
        config = load_config(raw)

        mock_task = MagicMock()
        mock_task.name = "t"
        mock_task.id = "i"
        mock_task.get_logger.return_value = MagicMock()
        with patch("clearml.Task.init", return_value=mock_task) as init:
            build_logger(config)
        assert init.call_args.kwargs["tags"] == ["timm", "lr=0.001"]


class TestBuildBatchTransform:
    def test_mixup_built_with_global_targets(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform
        from src.transforms.batch import MixUp

        config = load_config(_minimal_config())  # 'label' = classification (GLOBAL)
        runtime = RuntimeContext()
        runtime.num_classes["label"] = 3
        ctx = WiringContext(config=config, runtime=runtime)

        transform = _build_batch_transform({"name": "mixup", "alpha": 0.2}, ctx)

        assert isinstance(transform, MixUp)
        assert transform._targets[0].num_classes == 3  # injected from the task

    def test_mosaic_built_with_dense_targets(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform
        from src.transforms.batch import Mosaic

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        ctx = WiringContext(config=config, runtime=RuntimeContext())

        transform = _build_batch_transform({"name": "mosaic"}, ctx)

        assert isinstance(transform, Mosaic)

    def test_guard_rejects_mixup_with_dense_head(self) -> None:
        from src.composition.wiring import WiringContext
        from src.composition.wiring.callbacks import _build_batch_transform

        raw = _minimal_config()
        raw["tasks"] = {"mask": {"preset": "segmentation", "target": "mask_path", "num_classes": 4}}
        config = load_config(raw)
        ctx = WiringContext(config=config, runtime=RuntimeContext())

        with pytest.raises(ValueError, match="coherent target"):
            _build_batch_transform({"name": "mixup"}, ctx)

    def test_build_callbacks_wires_batch_transform(self) -> None:
        from src.composition.wiring import build_callbacks

        raw = _minimal_config()
        raw["callbacks"] = {
            "mixup": {"name": "batch_transform", "disable_after_fraction": 0.5, "transform": {"name": "mixup"}}
        }
        config = load_config(raw)
        runtime = RuntimeContext()
        runtime.num_classes["label"] = 3

        callbacks = build_callbacks(config, runtime)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], BatchTransformCallback)


class TestBuildTrainer:
    """``build_trainer`` is the single home for Trainer construction (profiler seam)."""

    @staticmethod
    def _cpu_trainer(**trainer_extra: Any) -> dict[str, Any]:
        return {"accelerator": "cpu", "devices": 1, **trainer_extra}

    def test_profiler_target_dict_is_instantiated(self) -> None:
        from lightning.pytorch.profilers import AdvancedProfiler

        from src.composition.wiring import build_trainer

        config = load_config(
            _minimal_config(
                trainer=self._cpu_trainer(
                    profiler={
                        "_target_": "lightning.pytorch.profilers.AdvancedProfiler",
                        "dirpath": "/tmp/prof",
                        "filename": "report",
                    }
                )
            )
        )
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), AdvancedProfiler)

    def test_profiler_string_alias_passes_through(self) -> None:
        from lightning.pytorch.profilers import SimpleProfiler

        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(trainer=self._cpu_trainer(profiler="simple")))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), SimpleProfiler)

    def test_profiler_none_disables_profiling(self) -> None:
        from lightning.pytorch.profilers import PassThroughProfiler

        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(trainer=self._cpu_trainer()))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert isinstance(getattr(trainer, "profiler"), PassThroughProfiler)

    def test_epochs_and_save_dir_forwarded(self) -> None:
        from src.composition.wiring import build_trainer

        config = load_config(_minimal_config(epochs=7, save_dir="/tmp/run", trainer=self._cpu_trainer()))
        trainer = build_trainer(config, logger=False, callbacks=[])
        assert trainer.max_epochs == 7
        assert str(trainer.default_root_dir) == "/tmp/run"

    def test_dataloader_block_maps_to_config(self) -> None:
        config = load_config(_minimal_config(dataloader={"num_workers": 3, "pin_memory": True}))
        assert config.dataloader.num_workers == 3
        assert config.dataloader.pin_memory is True

    def test_dataloader_extra_key_survives_validation(self) -> None:
        config = load_config(_minimal_config(dataloader={"timeout": 5}))
        assert config.dataloader.model_extra == {"timeout": 5}

    def test_dataloader_reserved_key_is_rejected(self) -> None:
        from src.config import ConfigError

        with pytest.raises(ConfigError, match="managed by the framework"):
            load_config(_minimal_config(dataloader={"shuffle": True}))


class TestArcFaceEmbeddingWiring:
    """Pins the arcface_embedding preset end-to-end: config dict → wired Task.

    Parts (encoder default, dimension seam, activation) landed in prior tasks and are
    unit-tested there; this integration test may already pass — that is expected, it is
    pinning the seam rather than driving new red/green cycles.
    """

    def test_embedder_task_wired_from_config(self, make_image_csv: Callable[..., Path]) -> None:
        from src.losses.angular import ProxyAngularCriterion
        from src.tasks.activations import NormalizeActivation

        csv_path = make_image_csv(count=15, size=32, seed=3)
        raw = _minimal_config()
        raw["data"]["sources"] = str(csv_path)
        raw["tasks"] = {"embed": {"preset": "arcface_embedding", "target": "label", "dim": 16}}
        config = load_config(raw)

        bindings = build_bindings(config)
        assert type(bindings[0].encoder).__name__ == "LabelEncoder"  # preset default_encoder="label"

        runtime = RuntimeContext()
        runtime.num_classes["embed"] = 3  # what DataModule.setup() would infer from the label column
        task = build_tasks(config, runtime)[0]
        assert task.head_spec.out_features == 16
        assert isinstance(task.activation, NormalizeActivation)
        assert isinstance(task.criterion, ProxyAngularCriterion)
        assert task.criterion.prototypes.shape == (16, 3)
