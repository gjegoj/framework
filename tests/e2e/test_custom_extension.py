"""Extending the framework without editing it: one class of your own, one `_target_`, one run.

This is the promise the whole arrangement exists for. Nothing below is registered anywhere and nothing
under ``src/`` knows these classes: a kind of task and a network arrive by import path, and every part
around them — encoders, the pipeline, the loss, the metrics, the loop — is assembled as usual.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor, nn

from src.build import build
from src.config import load_config
from src.core import Batch, ModelOutput, TargetInfo, TensorTree, drop_class_axis
from src.experiment import run
from src.models import Model
from src.tasks import Task
from src.tasks.base import LossDeclaration
from src.training import StandardLearner
from tests.support.declarations import NORMALIZATION, pixel_pipeline
from tests.support.table import write_table

SIZE = [8, 8]
HERE = "tests.e2e.test_custom_extension"


class Doubling(Task):
    """A kind of one's own: the column holds a number, and the model learns twice it.

    Declared entirely by what the framework asks of a task — which encoder reads the column, what the
    loss compares, what a metric scores, what the output means — and by nothing else.
    """

    default_target_encoder: ClassVar[str | None] = "scalar"
    default_metrics: ClassVar[Mapping[str, Mapping[str, object]]] = {"mae": {"name": "mae"}}

    @classmethod
    def out_features(cls, info: TargetInfo) -> int:
        return 1

    @property
    def default_loss(self) -> LossDeclaration:
        return "mae"

    def loss_target(self, batch: Batch) -> Tensor:
        return self.target(batch).float() * 2

    def metric_view(self, batch: Batch) -> Tensor:
        return self.loss_target(batch)

    def postprocess(self, output: ModelOutput) -> TensorTree:
        return drop_class_axis(self.raw(output))


class Tiny(Model):
    """A network of one's own, arriving whole: no backbone to compose heads onto, and none needed."""

    def __init__(self, task: str = "age") -> None:
        super().__init__()
        self.task = task
        self.weights = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        image = inputs["image"]
        assert isinstance(image, Tensor)
        return ModelOutput(outputs={self.task: self.weights(image).mean(dim=(1, 2, 3))})


@pytest.fixture(scope="module")
def table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_table(tmp_path_factory.mktemp("rows"))


def declaration(table: Path, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "seed": 0,
        "epochs": 1,
        "batch_size": 2,
        "data": {
            "name": "table",
            "source": str(table),
            "inputs": {"image": {"column": "image_path"}},
            "split": {"train": 0.5, "val": 0.5},
        },
        "preprocessing": {
            "name": "standard",
            "inputs": {"image": {"name": "image", "image_size": SIZE, **NORMALIZATION}},
        },
        "transforms": {stage: pixel_pipeline(SIZE) for stage in ("train", "val")},
        "model": {"name": "composite", "backbone": {"name": "timm", "model_name": "resnet18", "pretrained": False}},
        "tasks": {"age": {"kind": {"_target_": f"{HERE}.Doubling"}, "target_column": "age"}},
        "trainer": {
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "run": {"directory": str(tmp_path / "run"), "test": False},
        **overrides,
    }


class EveryOther:
    """A rule of one's own for dividing a table: odd rows to the first split, even rows to the rest.

    Nothing about splitting is registered anywhere, so this reaches a run the same way a network or a
    pixel chain does — by import path, with its own options.
    """

    def __init__(self, first: str) -> None:
        self.first = first

    def __call__(self, rows: Any, fractions: Mapping[str, float], seed: int) -> dict[str, Any]:
        rest = [name for name in fractions if name != self.first]
        return {self.first: rows.iloc[::2].reset_index(drop=True), rest[0]: rows.iloc[1::2].reset_index(drop=True)}


def test_a_rule_for_dividing_a_table_arrives_by_import_path_like_everything_else(table: Path, tmp_path: Path) -> None:
    """Splitting was the one decision with no seam; it is declared now like every other one."""
    divided = {"train": 0.5, "val": 0.5, "rule": {"_target_": f"{HERE}.EveryOther", "first": "train"}}
    declared = declaration(table, tmp_path)
    declared["data"] = {**declared["data"], "split": divided}

    experiment = build(load_config(declared))

    rows = sum(batch.count for batch in experiment.data.train_dataloader())

    assert rows == 4, "every other one of eight"


def test_a_kind_of_task_nobody_registered_is_assembled_like_any_other(table: Path, tmp_path: Path) -> None:
    declared = declaration(table, tmp_path)
    experiment = build(load_config(declared))

    run(experiment)

    learner = experiment.module.learner
    assert isinstance(learner, StandardLearner)
    assert isinstance(learner.tasks["age"], Doubling), "the kind the run named is the kind it got"
    assert type(learner.losses["age"]).__name__ == "MeanAbsoluteError"
    assert experiment.trainer.logged_metrics["train/age/mae"] >= 0


def test_a_network_that_arrives_whole_replaces_the_composed_family(table: Path, tmp_path: Path) -> None:
    """No backbone, no heads, no sizes to resolve — the model section names a class and that is all."""
    declared = declaration(table, tmp_path, model={"_target_": f"{HERE}.Tiny"})
    experiment = build(load_config(declared))
    model = experiment.module.learner.model
    assert isinstance(model, Tiny), "the model section named a class, and that is the model"
    before = model.weights.weight.clone()

    run(experiment)

    assert not torch.equal(model.weights.weight, before), "it trained"
