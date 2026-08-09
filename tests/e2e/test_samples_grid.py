"""End-to-end: a run with the samples callback ships a page a browser can open."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.assembly import assemble, run
from src.core.normalisation import IMAGENET_MEAN, IMAGENET_STD
from tests.support.configs import disk_config
from tests.support.fakes import PageLogger

if TYPE_CHECKING:
    from pathlib import Path

MEAN, STD = list(IMAGENET_MEAN), list(IMAGENET_STD)
"""What the shared transforms normalise by, so the page draws the pixels back."""


@pytest.mark.e2e
@pytest.mark.slow
def test_a_run_with_the_samples_callback_ships_a_self_contained_page(dataset_root: Path) -> None:
    """The whole path: step returned, annotated, rendered, logged — real config, real data.

    The unit tests each hold one link. This asserts they are joined: the callback
    is reachable by name from YAML, a step's return value survives Lightning's own
    routing to the batch-end hook, the dataset's readable cells reach the page, and
    what lands in the tracker is a page that needs no second request to display.
    """
    config = disk_config(
        dataset_root,
        loader={"batch_size": 2, "drop_last": True},
        callbacks=[{"name": "samples", "every_n_epochs": 1, "stages": ["val"], "mean": MEAN, "std": STD}],
        run={"directory": str(dataset_root / "run"), "train": True, "test": False},
    )
    experiment = assemble(config)
    logger = PageLogger()
    experiment.trainer._loggers = [logger]

    run(experiment, config)

    assert logger.pages, "the callback drew nothing on a due validation batch"
    title, page, _ = logger.pages[0]
    assert title == "samples/val"
    assert "label::gt::" in page
    assert "label::pred::" in page
    assert "images/" in page  # the dataset's own cell became the source pill
    assert "<style>" in page
    assert "<script src" not in page
