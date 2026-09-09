"""A table-declared detection dataset: canon rows in, letterboxed ``Instances`` batches out."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.core import Instances, Stage
from src.data.collate import collate_samples
from src.data.converters.annotations import annotation_object, annotation_row, write_annotations
from tests.support.configs import pipeline_of
from tests.support.detection import annotation_tree, detection_config


def test_canon_rows_become_letterboxed_instances_batches(tmp_path: Path) -> None:
    """Every seam of the stage at once: source, encoder, geometry, collate — one assertion each."""
    annotation_tree(tmp_path)
    module = pipeline_of(detection_config(tmp_path))
    facts = module.setup()

    loader = DataLoader(module.dataset(Stage.TRAIN), batch_size=3, collate_fn=collate_samples)
    batch = next(iter(loader))
    merged = batch.targets["boxes"]

    assert facts["boxes"].num_classes == 2
    assert facts["boxes"].class_names == ("cat", "dog")
    assert isinstance(merged, Instances)
    assert batch.inputs["image"].shape == (3, 3, 64, 64)


def test_the_boxes_of_a_batch_carry_the_letterbox_arithmetic_end_to_end(tmp_path: Path) -> None:
    """The number measured on the seam, now through encoder, transform and collation."""
    annotation_tree(tmp_path)
    module = pipeline_of(detection_config(tmp_path))
    module.setup()

    loader = DataLoader(module.dataset(Stage.VAL), batch_size=1, collate_fn=collate_samples)
    merged = next(iter(loader)).targets["boxes"]

    assert isinstance(merged, Instances)
    assert torch.allclose(merged.boxes, torch.tensor([[16.0, 24.0, 48.0, 40.0]]))
    assert merged.labels.tolist() == [1]  # "dog" is second in the learned vocabulary


def test_a_negative_row_takes_its_place_in_the_batch_holding_nothing(tmp_path: Path) -> None:
    """The row without objects is an image the model must learn to leave empty."""
    annotation_tree(tmp_path)
    module = pipeline_of(detection_config(tmp_path))
    module.setup()

    dataset = module.dataset(Stage.TRAIN)
    batch = collate_samples([dataset[index] for index in range(3)])
    merged = batch.targets["boxes"]

    assert isinstance(merged, Instances)
    assert len(merged.of(2).boxes) == 0
    assert merged.sample_index.tolist() == [0, 1]


def test_a_class_unknown_in_the_val_rows_is_refused_at_setup_naming_the_stage(tmp_path: Path) -> None:
    """Encoders fit on train and setup validates every other split against them, so a typo in
    ``val.jsonl`` dies before the first epoch rather than at the first validation ``encode``.
    """
    annotation_tree(tmp_path)
    wolf = [annotation_object((50.0, 25.0, 150.0, 75.0), "wolf")]
    write_annotations([annotation_row("d.jpg", wolf)], tmp_path / "val.jsonl")

    with pytest.raises(LookupError, match=r"val.*wolf"):
        pipeline_of(detection_config(tmp_path)).setup()
