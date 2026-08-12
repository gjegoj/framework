"""``auxiliary_inputs``: arrays only the augmentations read — nothing downstream sees them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import ExperimentConfig
from src.core.entities import Sample
from src.data.collate import collate_samples
from src.data.dataset import TableDataset
from src.data.loaders import MaskLoader
from src.data.schema import DataSchema, InputColumn


def test_a_sample_carries_auxiliary_inputs_apart_from_model_inputs() -> None:
    sample = Sample(inputs={"image": 1}, targets={}, auxiliary_inputs={"lesion": np.ones((2, 2))})

    assert "lesion" not in sample.inputs


def test_collation_leaves_auxiliary_inputs_behind() -> None:
    """The point of the field: no code drops it — the batch simply has no such slot."""
    samples = [
        Sample(inputs={"image": np.zeros((2, 2, 3))}, targets={"y": 1.0}, auxiliary_inputs={"lesion": np.ones((2, 2))})
        for _ in range(2)
    ]

    batch = collate_samples(samples)

    assert not hasattr(batch, "auxiliary_inputs")
    assert set(batch.inputs) == {"image"}
    assert set(batch.targets) == {"y"}


def test_the_dataset_loads_auxiliary_inputs_from_their_columns() -> None:
    table = pd.DataFrame({"pixels": [0], "mask_cell": ["m0"]})
    schema = DataSchema(
        inputs={"image": InputColumn(column="pixels", loader=lambda cell: np.zeros((4, 4, 3), np.uint8))},
        targets={},
        auxiliary_inputs={"lesion": InputColumn(column="mask_cell", loader=lambda cell: np.ones((4, 4), np.uint8))},
    )

    sample = TableDataset(table, schema, transform=None)[0]

    assert set(sample.auxiliary_inputs) == {"lesion"}
    assert sample.auxiliary_inputs["lesion"].shape == (4, 4)


def test_an_auxiliary_column_missing_from_the_table_fails_at_construction() -> None:
    schema = DataSchema(
        inputs={"image": InputColumn(column="pixels", loader=lambda cell: cell)},
        targets={},
        auxiliary_inputs={"lesion": InputColumn(column="absent", loader=lambda cell: cell)},
    )

    with pytest.raises(ValueError, match="absent"):
        TableDataset(pd.DataFrame({"pixels": [0]}), schema, transform=None)


def test_the_config_defaults_an_auxiliary_input_to_the_mask_loader() -> None:
    config = ExperimentConfig.model_validate(
        {
            "data": {
                "source": "rows.csv",
                "inputs": {"image": {"column": "image_path"}},
                "auxiliary_inputs": {"lesion": {"column": "mask_path"}},
            },
            "tasks": {"warmth": {"preset": "regression", "target": "warmth"}},
            "model": {"name": "timm", "model_name": "resnet18"},
        }
    )

    assert config.data.auxiliary_inputs["lesion"].loader.name == "mask"


def test_the_mask_loader_reads_one_plane_and_says_it_is_spatial(tmp_path: Path) -> None:
    """``spatial`` is the class-level marker assembly derives mask treatment from;
    ``grayscale`` alone cannot carry it — an X-ray is a grayscale *photograph*."""
    import cv2

    cv2.imwrite(str(tmp_path / "m.png"), np.eye(4, dtype=np.uint8))

    loaded = MaskLoader(root=tmp_path)("m.png")

    assert loaded.shape == (4, 4)
    assert MaskLoader.spatial is True


def test_assembly_offers_the_auxiliary_input_names_to_the_pipeline() -> None:
    """Never written by hand in config: derived from data.auxiliary_inputs, the same
    channel spatial_targets already travels."""
    from src.assembly.data import build_data_schema, build_transforms
    from src.core.taxonomy import Stage

    config = ExperimentConfig.model_validate(
        {
            "data": {
                "source": "rows.csv",
                "inputs": {"image": {"column": "image_path"}},
                "auxiliary_inputs": {"lesion": {"column": "mask_path"}},
            },
            "tasks": {"warmth": {"preset": "regression", "target": "warmth"}},
            "model": {"name": "timm", "model_name": "resnet18"},
            "transforms": {
                "train": {
                    "_target_": "src.transforms.AlbumentationsTransform",
                    "transforms": [{"_target_": "albumentations.HorizontalFlip", "p": 1.0}],
                }
            },
        }
    )

    built = build_transforms(config, build_data_schema(config))[Stage.TRAIN]
    sample = built(
        Sample(
            inputs={"image": np.zeros((4, 4, 3), np.uint8)},
            targets={},
            auxiliary_inputs={"lesion": np.eye(4, dtype=np.uint8)},
        )
    )

    flipped = np.asarray(sample.auxiliary_inputs["lesion"]).squeeze()
    assert np.array_equal(flipped, np.fliplr(np.eye(4, dtype=np.uint8)))


def test_a_masked_augmentation_feeds_a_binned_target_and_the_mask_dies_with_the_sample(tmp_path: Path) -> None:
    """The scenario both parts of this work exist for: warmth is drawn inside a mask,
    encoded into bins *after* the transforms, and the mask itself never reaches the batch.
    """
    import csv

    import cv2

    from src.assembly.data import build_data_schema, build_transforms
    from src.core.taxonomy import Stage
    from src.data.sources import CsvSource

    cv2.imwrite(str(tmp_path / "0.png"), np.full((32, 32, 3), 128, np.uint8))
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 1
    cv2.imwrite(str(tmp_path / "0_mask.png"), mask)
    with (tmp_path / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path", "warmth"])
        writer.writeheader()
        writer.writerow(
            {"image_path": str(tmp_path / "0.png"), "mask_path": str(tmp_path / "0_mask.png"), "warmth": 0.0}
        )

    config = ExperimentConfig.model_validate(
        {
            "data": {
                "source": str(tmp_path / "rows.csv"),
                "inputs": {"image": {"column": "image_path"}},
                "auxiliary_inputs": {"lesion": {"column": "mask_path"}},
            },
            "tasks": {
                "warmth": {
                    "preset": "regression",
                    "target": "warmth",
                    "target_encoder": {"name": "gaussian_bins", "bins": 8, "low": 3000, "high": 4600},
                }
            },
            "model": {"name": "timm", "model_name": "resnet18"},
            "transforms": {
                "train": {
                    "_target_": "src.transforms.AlbumentationsTransform",
                    "label_targets": ["warmth"],
                    "transforms": [
                        {
                            "_target_": "src.transforms.augmentations.MaskedPlanckianJitter",
                            "mask_key": "lesion",
                            "temperature_range": [3400, 4200],
                            "spread": 400,
                            "p": 1.0,
                        }
                    ],
                }
            },
        }
    )
    schema = build_data_schema(config)
    dataset = TableDataset(
        CsvSource(paths=[str(tmp_path / "rows.csv")]).read(),
        schema,
        build_transforms(config, schema)[Stage.TRAIN],
    )

    batch = collate_samples([dataset[0]])

    assert set(batch.inputs) == {"image"}  # the mask is nowhere in the batch

    warmth = batch.targets["warmth"]
    assert isinstance(warmth, torch.Tensor)  # a binned regression target collates to one
    assert tuple(warmth.shape) == (1, 8)  # a distribution, not the stub scalar
    assert float(warmth.sum()) == pytest.approx(1.0)

    # Shape and sum alone would also hold for the stub 0.0 — measured, it encodes to an
    # expectation of 1133 K. Reading the distribution back is what proves the
    # augmentation's own draw is what got encoded.
    centres = np.asarray(schema.targets["warmth"].encoder.class_values)
    expectation = float((warmth[0].numpy() * centres).sum())
    assert 3400 <= expectation <= 4200


def test_a_mask_loaded_model_input_is_marked_spatial_in_the_schema() -> None:
    """The marker is read off the raw loader BEFORE the cache wraps it: ``cached()``
    returns a bare closure, so probing the wrapped loader would silently read False."""
    from src.assembly.data import build_cache, build_data_schema

    config = ExperimentConfig.model_validate(
        {
            "data": {
                "source": "rows.csv",
                "inputs": {
                    "image": {"column": "image_path"},
                    "lesion_mask": {"column": "mask_path", "loader": {"name": "mask"}},
                },
                "cache": {"name": "ram", "max_gib": 1},
            },
            "tasks": {"warmth": {"preset": "regression", "target": "warmth"}},
            "model": {"name": "timm", "model_name": "resnet18"},
        }
    )

    schema = build_data_schema(config, build_cache(config))

    assert schema.inputs["lesion_mask"].spatial is True
    assert schema.inputs["image"].spatial is False


def test_a_mask_model_input_reaches_the_batch_uncorrupted(tmp_path: Path) -> None:
    """Image *and* mask into the model: collated like any input, resized with the image,
    and still binary afterwards.

    Both halves are needed to pin this. Measured on the three states the wiring can be
    in: **unregistered**, the key is never passed to the pipeline at all and comes back
    ``(16, 16)`` and binary — which "still binary" alone would have accepted;
    **image-kind**, it is resized but ``Normalize`` rewrites every value to
    ``-2.118..``; **mask-kind**, resized and untouched. The shape assertion rules out
    the first, the value assertion the second.
    """
    import csv

    import cv2

    from src.assembly.data import build_data_schema, build_transforms
    from src.core.taxonomy import Stage
    from src.data.sources import CsvSource

    cv2.imwrite(str(tmp_path / "0.png"), np.full((16, 16, 3), 128, np.uint8))
    cv2.imwrite(str(tmp_path / "0_mask.png"), (np.eye(16) > 0).astype(np.uint8))
    with (tmp_path / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "mask_path", "warmth"])
        writer.writeheader()
        writer.writerow(
            {"image_path": str(tmp_path / "0.png"), "mask_path": str(tmp_path / "0_mask.png"), "warmth": 1.0}
        )

    config = ExperimentConfig.model_validate(
        {
            "data": {
                "source": str(tmp_path / "rows.csv"),
                "inputs": {
                    "image": {"column": "image_path"},
                    "lesion_mask": {"column": "mask_path", "loader": {"name": "mask"}},
                },
            },
            "tasks": {"warmth": {"preset": "regression", "target": "warmth"}},
            "model": {"name": "timm", "model_name": "resnet18"},
            "transforms": {
                "train": {
                    "_target_": "src.transforms.AlbumentationsTransform",
                    "transforms": [
                        {"_target_": "albumentations.Resize", "height": 8, "width": 8},
                        {"_target_": "albumentations.Normalize"},
                    ],
                }
            },
        }
    )
    schema = build_data_schema(config)
    dataset = TableDataset(
        CsvSource(paths=[str(tmp_path / "rows.csv")]).read(),
        schema,
        build_transforms(config, schema)[Stage.TRAIN],
    )

    batch = collate_samples([dataset[0]])

    assert "lesion_mask" in batch.inputs  # a real model input, collated like any other

    carried = batch.inputs["lesion_mask"]
    assert tuple(carried.shape)[-2:] == (8, 8)  # it went through the pipeline
    assert set(carried.unique().tolist()) == {0, 1}  # as a mask: Normalize never saw it
