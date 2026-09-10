"""An experiment declares what; whatever the framework derives or owns is refused where it would be restated."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.config import ExperimentConfig, TaskConfig, load_config
from src.core import Stage
from tests.support.declarations import CLASSES


def test_the_minimal_experiment_fills_in_every_default(minimal: dict[str, Any]) -> None:
    config = load_config(minimal)

    assert (config.lr, config.batch_size, config.epochs, config.seed) == (1e-3, 16, 10, 42)
    assert config.learner.name == "standard" and config.optimizer.name == "adamw"
    assert config.scheduler is None and config.tracker is None and config.export == []
    assert config.run.train and config.run.test


@pytest.mark.parametrize(
    ("section", "value", "reason"),
    [
        pytest.param("tasks", {}, "at least one task", id="no tasks"),
        pytest.param("tasks", {"a/b": {"kind": "classification"}}, "Task", id="task name with a separator"),
        pytest.param("optimizer", {"name": "adamw", "lr": 1e-4}, "experiment.lr", id="lr on the optimizer"),
        pytest.param("loader", {"batch_size": 8}, "experiment.batch_size", id="batch size on the loader"),
        pytest.param("loader", {"shuffle": True}, "shuffle", id="shuffle is a stage convention"),
        pytest.param("trainer", {"max_epochs": 3}, "experiment.epochs", id="epochs on the trainer"),
        pytest.param("trainer", {"callbacks": []}, "experiment.callbacks", id="callbacks on the trainer"),
        pytest.param("model", {"name": "composite", "heads": {}}, "tasks", id="heads on the model"),
        pytest.param("unknown", 1, "unknown", id="unknown section"),
    ],
)
def test_refuses_a_value_declared_where_the_framework_owns_it(
    minimal: dict[str, Any], section: str, value: Any, reason: str
) -> None:
    with pytest.raises(ValidationError, match=reason):
        load_config({**minimal, section: value})


def test_forward_sections_pass_unknown_keys_through(minimal: dict[str, Any]) -> None:
    config = load_config({**minimal, "trainer": {"precision": "bf16-mixed"}, "loader": {"prefetch_factor": 4}})

    assert config.trainer.params == {"precision": "bf16-mixed"}
    assert config.loader.params == {"prefetch_factor": 4}


def test_child_positions_accept_registry_names_where_the_schema_declares_them(minimal: dict[str, Any]) -> None:
    """A schema-declared child (``model.backbone``, ``preprocessing.inputs``) may say ``name``; others need a target."""
    config = load_config(
        {
            **minimal,
            "model": {"name": "composite", "backbone": {"name": "timm", "model_name": "resnet18"}},
            "preprocessing": {
                "name": "standard",
                "inputs": {"image": {"name": "image", "image_size": [224, 224]}},
                "collator": "stack",
            },
        }
    )

    assert config.model.backbone is not None and config.model.backbone.spelled == "timm"
    assert config.model.params == {}
    assert config.preprocessing is not None and config.preprocessing.collator is not None
    assert config.preprocessing.inputs is not None and config.preprocessing.inputs["image"].params == {
        "image_size": [224, 224]
    }


def test_transforms_are_keyed_by_stage(minimal: dict[str, Any]) -> None:
    config = load_config({**minimal, "transforms": {"train": "augmented", "val": {"name": "resize", "size": 224}}})

    assert set(config.transforms) == {Stage.TRAIN, Stage.VAL}


class TestTask:
    @pytest.fixture
    def task(self) -> dict[str, Any]:
        return {"kind": "classification", "target": "species", "classes": CLASSES}

    def test_classes_accept_string_indices_from_yaml(self, task: dict[str, Any]) -> None:
        assert TaskConfig.model_validate({**task, "classes": {"0": "cat", "1": "dog"}}).classes == {0: "cat", 1: "dog"}

    def test_classes_may_come_from_a_file(self, task: dict[str, Any]) -> None:
        classes = TaskConfig.model_validate({**task, "classes": {"file": "classes.txt"}}).classes

        assert classes is not None and not isinstance(classes, dict) and classes.file == "classes.txt"

    @pytest.mark.parametrize(
        "classes",
        [
            pytest.param({1: "cat"}, id="not from zero"),
            pytest.param({0: "cat", "0": "dog"}, id="colliding spellings"),
            pytest.param({"x": "cat"}, id="non-numeric index"),
            pytest.param(["cat", "dog"], id="a list"),
        ],
    )
    def test_refuses_a_vocabulary_that_cannot_index_outputs(self, task: dict[str, Any], classes: Any) -> None:
        with pytest.raises(ValidationError):
            TaskConfig.model_validate({**task, "classes": classes})

    def test_head_and_output_are_two_ways_to_reach_the_model_and_exclude_each_other(self, task: dict[str, Any]) -> None:
        TaskConfig.model_validate({**task, "head": {"name": "linear", "input": "pooled"}})
        TaskConfig.model_validate({**task, "output": "logits"})
        with pytest.raises(ValidationError, match="mutually exclusive"):
            TaskConfig.model_validate({**task, "head": "linear", "output": "logits"})

    @pytest.mark.parametrize("inputs", [[], ["pooled", "pooled"], [" "]], ids=["empty", "duplicate", "blank"])
    def test_a_head_names_distinct_feature_streams(self, task: dict[str, Any], inputs: list[str]) -> None:
        with pytest.raises(ValidationError):
            TaskConfig.model_validate({**task, "head": {"name": "linear", "input": inputs}})

    def test_a_loss_list_separates_the_loss_from_its_weight(self, task: dict[str, Any]) -> None:
        losses = [{"loss": "cross_entropy"}, {"loss": {"name": "dice", "smooth": 1.0}, "weight": 0.5, "log_name": "d"}]

        config = TaskConfig.model_validate({**task, "loss": losses})

        assert isinstance(config.loss, list)
        assert [(one.loss.spelled, one.weight) for one in config.loss] == [("cross_entropy", 1.0), ("dice", 0.5)]

    @pytest.mark.parametrize(
        "losses",
        [
            pytest.param([], id="empty"),
            pytest.param([{"loss": "ce", "weight": 0.0}], id="no positive weight"),
            pytest.param(
                [{"loss": "ce", "log_name": "x"}, {"loss": "dice", "log_name": "x"}], id="duplicate log names"
            ),
            pytest.param([{"name": "ce", "weight": 1.0}], id="flat spelling collides with a loss argument"),
        ],
    )
    def test_refuses_a_loss_list_that_cannot_be_reported(self, task: dict[str, Any], losses: list[Any]) -> None:
        with pytest.raises(ValidationError):
            TaskConfig.model_validate({**task, "loss": losses})

    @pytest.mark.parametrize(("field", "value"), [("weight", 0.0), ("weight", -1.0), ("lr", 0.0)])
    def test_rates_and_weights_are_positive(self, task: dict[str, Any], field: str, value: float) -> None:
        with pytest.raises(ValidationError, match=field):
            TaskConfig.model_validate({**task, field: value})


class TestRun:
    def test_one_way_to_restore_weights(self, minimal: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="one of"):
            load_config({**minimal, "run": {"checkpoint_path": "a.ckpt", "resume_path": "b.ckpt"}})

    def test_resuming_implies_training(self, minimal: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="train"):
            load_config({**minimal, "run": {"resume_path": "b.ckpt", "train": False}})


def test_the_schema_is_a_leaf_that_names_nothing_it_builds() -> None:
    """Loading the schema resolves no implementation: a typo in a name dies at build, not at import."""
    assert ExperimentConfig.model_fields["model"].annotation is not None
