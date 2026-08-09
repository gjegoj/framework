"""A metric may compute several numbers at once, and a per-class one may be sparse.

Both shapes arrived with detection and neither is detection's alone: `MeanAveragePrecision`
returns fifteen keys, of which two are per-class readings aligned with a `classes` tensor
rather than with the task's class indices. What the grammar did before was raise on the
first and mislabel the second.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torchmetrics import Metric

from src.core import PerClass
from src.core.reporting import report_metric
from src.metrics import WrappedMetricSet


def logged_by(key: str, value: Any, class_names: list[str] | None = None) -> dict[str, float]:
    """What `report_metric` sent to the scalar log for one computed value."""
    written: dict[str, float] = {}
    report_metric(
        key,
        value,
        scalar_log=lambda name, number: written.__setitem__(name, float(number)),
        logger=None,
        step=0,
        class_names=class_names,
    )
    return written


class FamilyMetric(Metric):
    """A metric computing several numbers together, as the COCO evaluators do."""

    higher_is_better = True
    readings = ("map", "map_50")

    seen: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.add_state("seen", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predictions: Any, target: Any) -> None:
        self.seen = self.seen + 1

    def compute(self) -> dict[str, torch.Tensor]:
        return {"map": torch.tensor(0.9), "map_50": torch.tensor(1.0)}


def test_a_family_of_readings_logs_one_key_per_reading() -> None:
    """A metric computing several numbers is a namespace, not a new geometry.

    The grammar already puts `{stage}/{task}/{metric}/{leaf}` on one graph, so a family
    needs routing rather than a shape of its own — and each reading is then placed by
    its own geometry, which is what lets a family carry a per-class member.
    """
    written = logged_by("val/boxes/map", {"map": torch.tensor(0.9), "map_50": torch.tensor(1.0)})

    assert written == {"val/boxes/map/map": pytest.approx(0.9), "val/boxes/map/map_50": pytest.approx(1.0)}


def test_a_per_class_reading_is_named_by_the_classes_it_is_about() -> None:
    """Measured: `map_per_class` is aligned with a `classes` tensor listing only the
    classes that appeared — `[0, 2]` — so naming by position logs class 2's number under
    class 1's name. Silently, and with nothing in the chart to make it noticeable.
    """
    written = logged_by(
        "val/boxes/map/map_per_class",
        PerClass(values=torch.tensor([1.0, 0.8]), classes=torch.tensor([0, 2])),
        class_names=["cat", "dog", "truck"],
    )

    assert written["val/boxes/map/map_per_class/cat"] == pytest.approx(1.0)
    assert written["val/boxes/map/map_per_class/truck"] == pytest.approx(0.8)
    assert "val/boxes/map/map_per_class/dog" not in written


def test_a_dense_vector_still_names_every_class_by_position() -> None:
    """The per-class metrics that were here first keep working unchanged.

    A dense reading is the case where position *is* the class, so it flows through the
    same one naming path rather than a second one kept in step with it by hand.
    """
    written = logged_by("val/label/f1", torch.tensor([0.5, 0.7]), class_names=["cat", "dog"])

    assert written["val/label/f1/cat"] == pytest.approx(0.5)
    assert written["val/label/f1/dog"] == pytest.approx(0.7)
    assert written["val/label/f1/mean"] == pytest.approx(0.6)


def test_a_metric_keeps_the_label_it_was_registered_under() -> None:
    """`MetricCollection` flattens a family's keys to the top level and drops the label.

    The set then looks up a key that is not a metric and raises, and two entries of one
    metric collide on identical output keys — so the label a run declared has to be kept
    by whoever declared it.
    """
    metric_set = WrappedMetricSet({"map": FamilyMetric(), "map50": FamilyMetric()})
    metric_set.update(torch.zeros(1), torch.zeros(1))

    computed = metric_set.compute()

    assert set(computed) == {"map", "map50"}
    assert computed["map"]["map_50"] == pytest.approx(1.0)


def test_a_families_readings_are_named_so_a_checkpoint_can_watch_one() -> None:
    """`directions` is how a monitor learns which way is better, and it answers per key.

    A family publishes several keys, so naming only the metric would leave every reading
    but one unwatchable — and `monitor: val/boxes/map/map_50` would fail at fit start.
    """
    metric_set = WrappedMetricSet({"map": FamilyMetric()})
    metric_set.update(torch.zeros(1), torch.zeros(1))

    directions = metric_set.directions()

    assert directions["map/map"] is True
    assert directions["map/map_50"] is True
