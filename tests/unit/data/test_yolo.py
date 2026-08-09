"""``YoloDataModule``: a native YOLO layout behind the same port a table uses."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.core import Batch, DataModule, DataProfile, Instances, Modality, Stage
from src.data import YoloDataModule

pytest.importorskip("ultralytics", reason="the YOLO pipeline is an optional dependency")

CLASSES = {0: "cat", 1: "dog"}


def yolo_dataset(root: Path, stages: tuple[str, ...] = ("train", "val")) -> str:
    """The directory layout ultralytics expects, small enough to build in a test."""
    for stage in stages:
        (root / "images" / stage).mkdir(parents=True, exist_ok=True)
        (root / "labels" / stage).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            cv2.imwrite(str(root / "images" / stage / f"{index}.jpg"), np.full((32, 32, 3), 128, dtype=np.uint8))
            # Distinct centres: ultralytics drops duplicate boxes, which would hide the count.
            boxes = "\n".join(f"{index % 2} {0.2 + 0.2 * box} 0.5 0.1 0.1" for box in range(index + 1))
            (root / "labels" / stage / f"{index}.txt").write_text(boxes)
    descriptor = root / "data.yaml"
    descriptor.write_text(
        yaml.safe_dump({"path": str(root), **{stage: f"images/{stage}" for stage in stages}, "names": CLASSES})
    )
    return str(descriptor)


def module(root: Path, **kwargs: object) -> YoloDataModule:
    return YoloDataModule(data_yaml=yolo_dataset(root), image_size=32, batch_size=2, **kwargs)  # type: ignore[arg-type]


def ready(root: Path, **overrides: object) -> YoloDataModule:
    """A set-up module — what every test here starts from, since `setup` builds the datasets."""
    built = module(root, **overrides)
    built.setup(DataProfile())
    return built


def test_it_is_a_data_module_like_any_other(tmp_path: Path) -> None:
    assert isinstance(module(tmp_path), DataModule)


def test_the_classes_reach_the_profile_so_a_head_can_size_itself(tmp_path: Path) -> None:
    """The same route a table takes: facts in, head width out — no config restates them."""
    profile = DataProfile()

    module(tmp_path, task_name="objects").setup(profile)

    assert profile.facts("objects").num_classes == 2
    assert profile.facts("objects").class_names == ["cat", "dog"]


def test_every_stage_the_descriptor_declares_has_a_dataset(tmp_path: Path) -> None:
    """A descriptor carrying all three gets all three — none of them standing in for another."""
    declared = YoloDataModule(
        data_yaml=yolo_dataset(tmp_path, stages=("train", "val", "test")), image_size=32, batch_size=2
    )

    declared.setup(DataProfile())

    assert all(len(declared.dataset(stage)) == 2 for stage in Stage)  # type: ignore[arg-type]
    assert declared.dataset(Stage.TEST) is not declared.dataset(Stage.VAL)


def test_a_descriptor_without_a_test_split_reports_that_it_has_none(tmp_path: Path) -> None:
    """Standard YOLO practice, and this pipeline answers for it rather than papering over it.

    Serving the validation set under the test stage is a decision about what a test
    metric means, so it is not made here: `TrainingData` makes it once, for every
    pipeline, and says so. What this pipeline owes is a truthful answer about what
    it has — which is also what lets that decision be taken at all.
    """
    built = ready(tmp_path)

    with pytest.raises(LookupError, match="train, val"):
        built.dataset(Stage.TEST)


def test_it_reports_its_own_batching(tmp_path: Path) -> None:
    """Detection targets are ragged, so the framework's stacking collate cannot serve them."""
    built = ready(tmp_path)

    assert built.collate is not None


def test_the_batch_it_builds_keeps_boxes_addressable(tmp_path: Path) -> None:
    """Boxes are concatenated across the batch; ``batch_idx`` is what says whose they are.

    The count itself is not asserted: training-mode augmentation may drop a box
    that lands outside the frame, so only the correspondence is invariant.
    """
    built = ready(tmp_path)
    dataset = built.dataset(Stage.TRAIN)
    collate = built.collate
    assert collate is not None

    batch = collate([dataset[0], dataset[1]])

    found = batch.targets["detection"]
    assert isinstance(found, Instances)
    assert len(found.boxes) == len(found.labels) == len(found.sample_index)
    assert set(found.sample_index.tolist()) <= {0, 1}


def test_the_batch_is_the_framework_currency_not_a_vendor_dict(tmp_path: Path) -> None:
    """Translating here is what keeps the model port from learning a second shape."""
    built = ready(tmp_path)
    collate = built.collate
    assert collate is not None

    batch = collate([built.dataset(Stage.VAL)[0]])

    assert isinstance(batch, Batch)
    assert "im_file" in batch.meta  # whatever ultralytics carries beyond the tensors


def test_ragged_targets_are_why_the_framework_collate_cannot_serve(tmp_path: Path) -> None:
    """Two images with different box counts have nothing to stack — hence the port property."""
    built = ready(tmp_path)

    counts = {built.dataset(Stage.VAL)[index]["bboxes"].shape[0] for index in range(2)}  # type: ignore[index]

    assert len(counts) > 1


def test_datasets_before_setup_are_refused(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="setup"):
        module(tmp_path).dataset(Stage.TRAIN)


def test_the_images_leave_as_the_floats_every_other_pipeline_hands_over(tmp_path: Path) -> None:
    """The framework's currency is a normalised float tensor.

    Left as the `uint8` ultralytics yields, dividing by 255 becomes the model's job — a
    transform living inside a model, and every later consumer of that batch inheriting
    the surprise. Their own trainer does it in `preprocess_batch`, which this contour
    replaces, so the conversion belongs to whoever replaced it.
    """
    built = ready(tmp_path)
    collate = built.collate
    assert collate is not None

    batch = collate([built.dataset(Stage.VAL)[0]])

    image = batch.inputs[Modality.IMAGE]
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0


def test_the_targets_are_one_entity_under_the_tasks_own_name(tmp_path: Path) -> None:
    """`Batch.targets` is keyed by task name everywhere else in the framework.

    Three loose top-level keys would make detection the one place a consumer has to
    learn a second shape before it can look anything up — and `targets["boxes"]` would
    collide with a task actually called `boxes`.
    """
    built = ready(tmp_path, task_name="objects")
    collate = built.collate
    assert collate is not None

    batch = collate([built.dataset(Stage.VAL)[0]])

    assert list(batch.targets) == ["objects"]
    assert isinstance(batch.targets["objects"], Instances)


def test_the_boxes_arrive_in_pixels_of_the_image_as_fed(tmp_path: Path) -> None:
    """Ultralytics stores normalised cxcywh; metrics compare xyxy pixels.

    Converting here, on the side that knows the letterboxed size, is what keeps a second
    copy of that size from being handed to the model — where the two could disagree and
    nothing would notice.
    """
    built = ready(tmp_path)
    collate = built.collate
    assert collate is not None

    batch = collate([built.dataset(Stage.VAL)[0]])

    found = batch.targets["detection"]
    assert isinstance(found, Instances)
    # The fixture writes one 0.1 x 0.1 box per image at y = 0.5, on a 32 px square.
    left, top, right, bottom = found.boxes[0].tolist()
    assert (right - left, bottom - top) == pytest.approx((3.2, 3.2), abs=1e-3)
    assert 0.0 <= left and right <= 32.0


def test_the_ultralytics_task_is_reachable_rather_than_shadowed(tmp_path: Path) -> None:
    """Ours names the profile key; theirs selects detect / segment / pose.

    Sharing one word means a segment run cannot say it is one: the framework would
    rename its own task and ultralytics would go on building a detector.
    """
    built = module(tmp_path, task_name="objects", task="segment")

    assert built.hyperparameters["task"] == "segment"
