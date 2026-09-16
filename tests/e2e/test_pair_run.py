"""A picture and its caption, pulled towards each other: two towers, a head apiece, one objective.

Assembled through the composition root, because what the thread is about is four declarations meeting —
two inputs, two networks side by side, one head built once per stream, and an objective that recovers
the pair from the rows it was handed. Nothing short of a real run puts the four in line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from src.build import build
from src.config import load_config
from src.core import require_tensor
from src.experiment import run
from src.models import CompositeModel, LinearHead
from src.models.heads import StackedHeads
from tests.support.declarations import BACKBONE, SIZE, smallest_run
from tests.support.table import write_table
from tests.support.text import WIDTH as FAMILY_WIDTH
from tests.support.text import text_family

if TYPE_CHECKING:
    from pathlib import Path

TASK, EMBEDDING, LENGTH, TOWERS = "pair", 12, 8, 2
IMAGE_WIDTH = 96
"""What the image tower pools to, against the text family's own 16: the two are deliberately unalike."""


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("pets"))


@pytest.fixture
def pairing(table: Path, tmp_path: Path) -> dict[str, Any]:
    """The shipped skeleton with the caption column put beside the picture and both read by a tower."""
    family = str(text_family(tmp_path / "family"))
    declared = dict(smallest_run(table, tmp_path))
    declared["data"] = {
        **declared["data"],
        "inputs": {"image": {"column": "image_path"}, "text": {"column": "caption"}},
    }
    declared["preprocessing"] = {
        **declared["preprocessing"],
        "inputs": {
            **declared["preprocessing"]["inputs"],
            "text": {"name": "text", "model_name": family, "max_length": LENGTH},
        },
    }
    declared["model"] = {
        "name": "composite",
        "backbone": {
            "_target_": "src.models.MultiEncoderBackbone",
            "encoders": {
                "image": {"_target_": "src.models.TimmBackbone", "model_name": BACKBONE, "pretrained": False},
                "text": {"_target_": "src.models.HFTextBackbone", "model_name": family},
            },
        },
    }
    declared["tasks"] = {
        TASK: {
            "kind": {"name": "contrastive", "embedding_dim": EMBEDDING},
            "head": {"name": "linear", "stream": ["image_pooled", "text_pooled"]},
        }
    }
    return declared


def test_a_run_pairing_two_inputs_trains_and_reports_what_it_descends(pairing: dict[str, Any]) -> None:
    """Two towers of different families and widths, and the objective over them is the one views use."""
    experiment = build(load_config(pairing))

    run(experiment)

    logged = experiment.module.trainer.logged_metrics
    assert f"test/{TASK}/info_nce" in logged
    assert float(logged["test/loss"]) == pytest.approx(float(logged[f"test/{TASK}/info_nce"]))


def test_each_tower_is_read_by_a_head_of_its_own_sized_by_what_that_tower_publishes(
    pairing: dict[str, Any],
) -> None:
    """Neither width is written in the declaration, and the two are not the same number."""
    model = build(load_config(pairing)).module.learner.model

    assert isinstance(model, CompositeModel)
    head = model.heads[TASK]
    assert isinstance(head, StackedHeads)
    towers = [part for part in head.heads.values() if isinstance(part, LinearHead)]
    assert [one.projection.in_features for one in towers] == [IMAGE_WIDTH, FAMILY_WIDTH]


def test_a_sample_answers_once_from_each_tower_and_its_answers_stay_next_to_each_other(
    pairing: dict[str, Any],
) -> None:
    """What the objective reads: as many rows as there are answers, a sample's own adjacent and unalike."""
    model = build(load_config(pairing)).module.learner.model
    samples = 3

    answered = model(
        {
            "image": torch.zeros(samples, 3, *SIZE),
            # A tree, as a tokenizer hands one over: the mask says which positions are words, and
            # the pooling that reads it refuses a caption arriving without one.
            "text": {
                "input_ids": torch.ones(samples, LENGTH, dtype=torch.long),
                "attention_mask": torch.ones(samples, LENGTH, dtype=torch.long),
            },
        }
    ).outputs[TASK]

    published = require_tensor(answered, name=TASK)
    assert tuple(published.shape) == (samples * TOWERS, EMBEDDING)
    assert not torch.equal(published[0], published[1])
