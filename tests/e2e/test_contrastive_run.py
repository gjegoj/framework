"""A run with no labels at all: what supervises it is that two draws came from the same picture.

Assembled through the composition root, because what the phase is about is three declarations meeting —
a stage that draws views, a backbone that folds them into the batch, and an objective that recovers the
pairs from the rows. Nothing short of a real run puts the three in line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.core import require_tensor
from src.experiment import run
from tests.support.declarations import SIZE, pixel_pipeline, smallest_run
from tests.support.table import write_table

if TYPE_CHECKING:
    from pathlib import Path

TASK, VIEWS, WIDTH = "appearance", 2, 16


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def contrasting(table: Path, tmp_path: Path) -> dict[str, Any]:
    """The shipped skeleton with every label taken out of it and views put in their place."""
    declared = dict(smallest_run(table, tmp_path))
    declared["transforms"] = {
        stage: {"_target_": "src.transforms.MultiViewTransform", "views": VIEWS, "base": pixel_pipeline(SIZE)}
        for stage in ("train", "val", "test")
    }
    declared["model"] = {
        "name": "composite",
        "backbone": {
            "_target_": "src.models.MultiViewBackbone",
            "backbone": {"_target_": "src.models.TimmBackbone", "model_name": "resnet18", "pretrained": False},
        },
    }
    declared["tasks"] = {TASK: {"kind": {"name": "contrastive", "embedding_dim": WIDTH}}}
    return declared


def test_a_run_that_declares_no_target_column_at_all_trains_and_reports_what_it_descends(
    contrasting: dict[str, Any],
) -> None:
    """The one kind whose supervision is the batch: no column is read, and a number is still reported.

    Read off the stage the run ends on, which is what ``logged_metrics`` keeps; that the total is made
    of this one term is the other half of the statement, since nothing else was declared to descend.
    """
    experiment = build(load_config(contrasting))

    run(experiment)

    logged = experiment.module.trainer.logged_metrics
    assert f"test/{TASK}/info_nce" in logged
    assert float(logged["test/loss"]) == pytest.approx(float(logged[f"test/{TASK}/info_nce"]))


def test_what_the_network_answers_with_is_one_direction_for_every_draw(contrasting: dict[str, Any]) -> None:
    """Views ride in the batch: the answers are as many as the draws, and each is a unit vector."""
    model = build(load_config(contrasting)).module.learner.model

    answered = model({"image": torch.zeros(3, VIEWS, 3, *SIZE)}).outputs[TASK]

    published = require_tensor(answered, name=TASK)
    assert tuple(published.shape) == (3 * VIEWS, WIDTH)


def test_the_scale_the_objective_learns_is_written_down_with_the_run(contrasting: dict[str, Any]) -> None:
    """A temperature learned and then forgotten is a model that cannot be read back the way it trained."""
    learner = build(load_config(contrasting)).module.learner

    assert f"losses.{TASK}.log_scale" in learner.state_dict()
