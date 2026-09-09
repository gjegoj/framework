"""End to end: a model declared outside the framework — its own head, its own loss — trains on a CSV.

The promise behind ``model: {_target_: ...}``: a model that arrives whole is built as it is,
fed by the table pipeline, judged by the task's metrics, and needs no registry to join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.build import build, run
from tests.support.configs import disk_config
from tests.support.fakes import OwnLossModel

if TYPE_CHECKING:
    from pathlib import Path

WHOLE = {"_target_": "tests.support.fakes.OwnLossModel", "num_classes": 2}
"""A model owning its head and its loss; what it needs it declares, since nothing composes it."""


def test_a_model_that_owns_its_loss_trains_on_a_csv_by_target(dataset_root: Path) -> None:
    """Its loss part logs under the framework's grammar, and the kind's default metrics still judge it."""
    config = disk_config(
        dataset_root, model=WHOLE, run={"directory": str(dataset_root / "run"), "train": True, "test": True}
    )
    experiment = build(config)

    run(experiment, config)

    assert isinstance(experiment.module.model, OwnLossModel)
    logged = set(experiment.trainer.callback_metrics)
    assert "test/label/ce" in logged  # the model's own part, scoped by the task
    assert {key for key in logged if key.startswith("test/label/") and key != "test/label/ce"}  # the kind's metrics
