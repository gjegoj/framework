"""``map``: mean average precision over ``Instances``, publishing what it was asked for."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Instances, PerClass
from src.metrics.detection import DEFAULT_READINGS, MeanAveragePrecisionOverInstances


def found(*rows: tuple[float, float, float, float, int, int, float]) -> Instances:
    """Detected objects: box, class, image, score."""
    return Instances(
        boxes=torch.tensor([row[:4] for row in rows], dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor([row[4] for row in rows], dtype=torch.int64),
        sample_index=torch.tensor([row[5] for row in rows], dtype=torch.int64),
        scores=torch.tensor([row[6] for row in rows], dtype=torch.float32),
    )


def truth(*rows: tuple[float, float, float, float, int, int]) -> Instances:
    """Annotated objects: box, class, image — and no confidence, because there is none."""
    return Instances(
        boxes=torch.tensor([row[:4] for row in rows], dtype=torch.float32).reshape(-1, 4),
        labels=torch.tensor([row[4] for row in rows], dtype=torch.int64),
        sample_index=torch.tensor([row[5] for row in rows], dtype=torch.int64),
    )


def computed(metric: MeanAveragePrecisionOverInstances, predictions: Instances, target: Instances) -> dict[str, Any]:
    metric.update(predictions, target)
    return metric.compute()


def test_a_perfect_prediction_scores_one() -> None:
    """The Instances-to-torchmetrics conversion is the whole of this class, so it is
    checked against an answer that cannot arise by accident: identical boxes score 1.
    """
    exact = computed(
        MeanAveragePrecisionOverInstances(),
        found((10, 10, 50, 50, 0, 0, 0.9)),
        truth((10, 10, 50, 50, 0, 0)),
    )

    assert float(exact["map"]) == pytest.approx(1.0)


def test_objects_are_split_back_to_the_images_they_belong_to() -> None:
    """torchmetrics compares per image.

    Flattened into one list, a box found in image 0 would be scored against image 1's
    annotation — and the number would be quietly wrong, high or low depending on the
    batch, which is the worst kind of wrong.
    """
    mixed = computed(
        MeanAveragePrecisionOverInstances(),
        found((10, 10, 50, 50, 0, 0, 0.9), (10, 10, 50, 50, 0, 1, 0.9)),
        truth((10, 10, 50, 50, 0, 0), (90, 90, 99, 99, 0, 1)),
    )

    assert 0.0 < float(mixed["map"]) < 1.0


def test_only_the_asked_for_readings_are_published() -> None:
    """Measured: `compute()` carries fifteen keys, of which three are what a detection run
    is read by. All fifteen on a graph from the first run is not a report.
    """
    published = computed(
        MeanAveragePrecisionOverInstances(),
        found((10, 10, 50, 50, 0, 0, 0.9)),
        truth((10, 10, 50, 50, 0, 0)),
    )

    assert list(published) == list(DEFAULT_READINGS) == ["map", "map_50", "map_75"]


def test_a_wider_selection_reaches_the_same_single_pass() -> None:
    """Asking for more costs nothing extra: torchmetrics computes them together, so a
    second reading is a second key from one result rather than a second accumulation.
    """
    wider = computed(
        MeanAveragePrecisionOverInstances(readings=["map", "map_small"]),
        found((10, 10, 20, 20, 0, 0, 0.9)),
        truth((10, 10, 20, 20, 0, 0)),
    )

    assert list(wider) == ["map", "map_small"]


def test_a_reading_that_is_not_applicable_is_dropped_rather_than_logged() -> None:
    """`-1.0` is torchmetrics' "no such object in this split" sentinel — measured on
    `map_large` with only small boxes. mAP lives in [0, 1], so logging it would put an
    impossible number on a chart and let a checkpoint monitor rank one.
    """
    small_only = computed(
        MeanAveragePrecisionOverInstances(readings=["map", "map_large"]),
        found((10, 10, 20, 20, 0, 0, 0.9)),
        truth((10, 10, 20, 20, 0, 0)),
    )

    assert "map_large" not in small_only
    assert "map" in small_only


def test_asking_for_a_per_class_reading_is_what_turns_its_cost_on() -> None:
    """`class_metrics` is how torchmetrics is told to produce `map_per_class`.

    Restating it in config would let the two disagree: a run asking for the reading and
    not paying for it receives the scalar -1 and no explanation of why.
    """
    assert MeanAveragePrecisionOverInstances(readings=["map"]).inner.class_metrics is False
    assert MeanAveragePrecisionOverInstances(readings=["map_per_class"]).inner.class_metrics is True


def test_a_per_class_reading_carries_the_classes_it_is_about() -> None:
    """Measured: `classes` is `[0, 2]` while `map_per_class` is `[1.0, 0.8]`, so the second
    number is class 2's. Handed on as a bare vector it would be named by position, and
    class 2's number would be logged under class 1's name.
    """
    per_class = computed(
        MeanAveragePrecisionOverInstances(readings=["map_per_class"]),
        found((10, 10, 50, 50, 0, 0, 0.9), (60, 60, 90, 90, 2, 0, 0.8)),
        truth((10, 10, 50, 50, 0, 0), (61, 61, 91, 91, 2, 0)),
    )

    reading = per_class["map_per_class"]
    assert isinstance(reading, PerClass)
    assert reading.classes.tolist() == [0, 2]


def test_the_backend_that_is_installed_is_the_one_used() -> None:
    """torchmetrics defaults to `pycocotools` and raises for it at compute time even where
    `faster-coco-eval` is the installed backend — so a declared dependency would look
    broken, an epoch into a run rather than at its start.

    Asserted through what the two do rather than through a private attribute: the default
    cannot compute here, and ours can.
    """
    from torchmetrics.detection import MeanAveragePrecision

    predictions, target = found((10, 10, 50, 50, 0, 0, 0.9)), truth((10, 10, 50, 50, 0, 0))
    default = MeanAveragePrecision(box_format="xyxy")
    default.update(_as_dicts(predictions), _as_dicts(target))

    with pytest.raises(ModuleNotFoundError, match="pycocotools"):
        default.compute()

    assert float(computed(MeanAveragePrecisionOverInstances(), predictions, target)["map"]) == pytest.approx(1.0)


def _as_dicts(objects: Instances) -> list[dict[str, Any]]:
    """The same conversion, spelled out here so the comparison is against torchmetrics itself."""
    entry: dict[str, Any] = {"boxes": objects.boxes, "labels": objects.labels}
    if objects.scores is not None:
        entry["scores"] = objects.scores
    return [entry]


def test_it_names_the_readings_it_publishes() -> None:
    """A checkpoint monitor is configured before a run computes anything, so which keys
    will exist has to be answerable from the metric itself.
    """
    assert MeanAveragePrecisionOverInstances(readings=["map", "map_50"]).readings == ("map", "map_50")


def test_a_reading_the_metric_does_not_compute_is_refused_by_name() -> None:
    """A typo would otherwise surface as a missing key at the end of the first epoch,
    an hour into a run, with nothing naming what was misspelt.
    """
    with pytest.raises(ValueError, match="map_50"):
        MeanAveragePrecisionOverInstances(readings=["map_fifty"])


def test_it_refuses_a_shape_it_cannot_compare() -> None:
    """A `map` on a classification task would otherwise fail inside torchmetrics, at a
    frame naming neither the task nor the metric.
    """
    with pytest.raises(TypeError, match="preset is 'detection'"):
        MeanAveragePrecisionOverInstances().update(torch.zeros(4, 3), torch.zeros(4))
