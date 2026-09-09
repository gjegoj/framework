"""Mosaic: four samples, one picture, and targets that follow the seams exactly."""

from __future__ import annotations

import pytest
import torch

from src.core import Batch, TaskFacts
from src.tasks import Classification, Detection, MetricLearning, Regression, Segmentation, Task, TaskKind
from src.transforms.batch import Mosaic
from tests.support.entities import a_task
from tests.support.narrowing import tensor


def task(name: str, kind: TaskKind | None = None, classes: int | None = 3) -> Task:
    return a_task(name=name, kind=kind, facts=TaskFacts(num_classes=classes))


def batch(size: int = 4, side: int = 8) -> Batch:
    """Every sample is one constant value, so a pixel names the sample it came from."""
    per_sample = torch.arange(1, size + 1, dtype=torch.float32).reshape(size, 1, 1, 1)
    return Batch(
        inputs={"image": per_sample.expand(size, 3, side, side).clone()},
        targets={"label": torch.arange(size) % 3},
    )


def test_every_pixel_keeps_exactly_one_source() -> None:
    """That is the difference from blending, and what lets a mask compose."""
    torch.manual_seed(0)
    given = batch()

    stitched = Mosaic().for_tasks([task("label", Classification())])(given)

    assert set(stitched.inputs["image"].flatten().tolist()) <= {1.0, 2.0, 3.0, 4.0}


def test_each_quadrant_comes_from_the_neighbour_that_far_along() -> None:
    """A quadrant taken from the wrong roll would desync the picture from the label."""
    torch.manual_seed(0)
    given = batch()
    per_sample = given.inputs["image"][:, 0, 0, 0]

    stitched = Mosaic().for_tasks([task("label", Classification())])(given)
    first = stitched.inputs["image"][0, 0]
    corners = (first[0, 0], first[0, -1], first[-1, 0], first[-1, -1])

    # Quadrant k comes from the batch rolled by k, and ``roll(k)`` brings sample ``i - k``.
    assert [float(corner) for corner in corners] == [float(per_sample.roll(k, 0)[0]) for k in range(4)]


def test_a_mask_is_swapped_by_the_same_seams_as_the_picture() -> None:
    """An index map has no meaningful average, so it is composed, never weighted."""
    torch.manual_seed(0)
    given = batch()
    given.targets["mask"] = torch.arange(4).reshape(4, 1, 1).expand(4, 8, 8).clone()
    tasks = [task("mask", Segmentation(), classes=4)]

    stitched = Mosaic().for_tasks(tasks)(given)

    assert tensor(stitched.targets["mask"]).dtype == tensor(given.targets["mask"]).dtype
    picture = stitched.inputs["image"][0, 0]
    assert torch.equal(tensor(stitched.targets["mask"])[0], (picture - 1).long())


def test_a_global_label_is_weighted_by_the_four_quadrant_areas() -> None:
    """The sample genuinely belongs to four classes, in the proportions shown."""
    torch.manual_seed(0)
    given = batch()

    stitched = Mosaic().for_tasks([task("label", Classification())])(given)

    assert tensor(stitched.targets["label"]).shape == (4, 3)
    assert torch.allclose(tensor(stitched.targets["label"]).sum(dim=1), torch.ones(4))


def test_the_label_reports_the_share_the_picture_actually_shows() -> None:
    """Weights the areas do not support would teach a class that is barely there."""
    torch.manual_seed(0)
    given = batch(size=4)
    given.targets["label"] = torch.arange(4)  # four distinct classes, one per sample

    stitched = Mosaic().for_tasks([task("label", Classification(), classes=4)])(given)
    picture = stitched.inputs["image"][0, 0]
    shown = torch.tensor([float((picture == value).sum()) / picture.numel() for value in (1, 2, 3, 4)])

    assert torch.allclose(tensor(stitched.targets["label"])[0], shown)


def test_a_continuous_target_is_weighted_as_the_number_it_already_is() -> None:
    given = batch()
    given.targets["price"] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    stitched = Mosaic().for_tasks([task("price", Regression(), classes=None)])(given)

    assert tensor(stitched.targets["price"]).shape == (4,)


def test_both_kinds_of_task_are_served_by_one_split() -> None:
    """The case Mosaic is best at: a mask and a label composed by the same seams."""
    torch.manual_seed(0)
    given = batch()
    given.targets["mask"] = torch.zeros(4, 8, 8, dtype=torch.long)
    tasks = [task("label", Classification()), task("mask", Segmentation(), classes=2)]

    stitched = Mosaic().for_tasks(tasks)(given)

    assert tensor(stitched.targets["label"]).shape == (4, 3)
    assert tensor(stitched.targets["mask"]).shape == (4, 8, 8)


def test_the_batch_it_was_given_is_left_alone() -> None:
    """The callback owns writing back; a transform that mutated would do it twice."""
    given = batch()
    before = given.inputs["image"].clone()

    Mosaic().for_tasks([task("label", Classification())])(given)

    assert torch.equal(given.inputs["image"], before)


def test_a_metric_learning_task_is_refused() -> None:
    """Proxy and margin losses break on soft labels."""
    metric = task("pairs", MetricLearning())

    with pytest.raises(ValueError, match="pairs"):
        Mosaic().for_tasks([metric])


def test_a_task_whose_targets_mosaic_cannot_rewrite_is_refused() -> None:
    """A detection task's boxes are not rewritten by stitching; refusing beats corrupting."""
    with pytest.raises(ValueError, match="pairs"):
        Mosaic().for_tasks([task("pairs", Detection())])


@pytest.mark.parametrize("split_range", [(0.0, 0.7), (0.7, 0.3), (0.3, 1.0)])
def test_a_split_that_cannot_make_four_quadrants_is_refused(split_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="split_range"):
        Mosaic(split_range=split_range).for_tasks([task("label", Classification())])
