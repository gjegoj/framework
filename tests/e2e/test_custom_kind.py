"""End to end: a kind of task declared outside the framework trains and draws like a shipped one.

The promise behind ``kind``: a new task is one class, reachable by ``_target_``, with no
edit to the framework. The class under test lives in ``tests/support/kinds.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.build import build, run
from src.config.experiment import IMAGENET_MEAN, IMAGENET_STD
from tests.support.configs import TASK, disk_config
from tests.support.fakes import PageLogger

if TYPE_CHECKING:
    from pathlib import Path

CUSTOM = {"label": TASK | {"kind": {"_target_": "tests.support.kinds.FocalClassification"}}}
"""The shipped classification task, its kind swapped for one the framework has never heard of."""


def test_a_kind_declared_by_target_trains_and_draws(dataset_root: Path) -> None:
    """Its loss, its metrics and its drawing all reach the run: the parts a kind states are the parts used."""
    config = disk_config(
        dataset_root,
        tasks=CUSTOM,
        loader={"batch_size": 2, "drop_last": True},
        callbacks=[
            {
                "name": "samples",
                "every_n_epochs": 1,
                "stages": ["val"],
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
            }
        ],
        run={"directory": str(dataset_root / "run"), "train": True, "test": True},
        logger={"_target_": "tests.support.fakes.PageLogger"},
    )
    experiment = build(config)
    logger = experiment.trainer.logger
    assert isinstance(logger, PageLogger)

    run(experiment, config)

    # ``trainer.test`` resets ``callback_metrics``, so what is left is the test pass — one stage is enough.
    logged = set(experiment.trainer.callback_metrics)
    assert "test/label/focal" in logged  # the kind's own loss, under the part it names
    assert "test/label/accuracy" in logged  # the kind's own default metrics, nothing of the shipped set
    assert not {key for key in logged if key.startswith("test/label/f1")}
    assert logger.pages, "the callback drew nothing on a due validation batch"
    assert "label::pred::" in logger.pages[0][1]  # drawn the way the parent kind draws
