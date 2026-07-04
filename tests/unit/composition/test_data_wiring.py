"""Data-layer wiring: source-config grammar, source bindings, staged sources, transforms, data sources."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from src.composition.wiring import (
    build_data_source,
    build_tasks,
    build_transforms,
)
from src.config import load_config
from src.core.enums import Stage
from src.core.runtime import RuntimeContext
from src.data.sources import CsvDataSource, JsonDataSource
from tests.support.builders import RESIZE_NORMALIZE_TOTENSOR as _RESIZE_NORMALIZE_TOTENSOR
from tests.support.builders import minimal_config as _minimal_config


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

        from src.core.entities import Sample

        config = load_config(_minimal_config())
        transform = build_transforms(config)[Stage.TRAIN]
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        sample = Sample(inputs={"image": image})
        result = transform.apply(sample)
        assert result.inputs["image"].shape == torch.Size([3, 64, 64])

    def test_config_transforms_used_when_present(self) -> None:

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
