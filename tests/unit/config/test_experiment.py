"""An experiment declares what; whatever the framework derives or owns is refused where it would be restated."""

from __future__ import annotations

from pathlib import Path
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
    assert config.scheduler is None and config.tracker is None and config.callbacks == []
    assert config.run.train and config.run.test


@pytest.mark.parametrize(
    ("section", "value", "reason"),
    [
        pytest.param("tasks", {}, "at least one task", id="no tasks"),
        pytest.param("tasks", {"a/b": {"kind": "classification"}}, "Task", id="task name with a separator"),
        pytest.param("optimizer", {"name": "adamw", "lr": 1e-4}, "the root's lr", id="lr on the optimizer"),
        pytest.param("loader", {"batch_size": 8}, "the root's batch_size", id="batch size on the loader"),
        pytest.param("loader", {"shuffle": True}, "not a declaration", id="shuffle is settled by the stage"),
        pytest.param(
            "callbacks",
            [{"name": "lr_monitor", "logging_interval": "epoch"}],
            "nowhere to write a learning rate",
            id="a rate watched with nothing recording",
        ),
        pytest.param("trainer", {"max_epochs": 3}, "the root's epochs", id="epochs on the trainer"),
        pytest.param("trainer", {"callbacks": []}, "the root's callbacks", id="callbacks on the trainer"),
        pytest.param("model", {"name": "composite", "heads": {}}, "tasks", id="heads on the model"),
        pytest.param(
            "tasks",
            {"t": {"kind": "classification", "loss": [{"loss": "cross_entropy", "log_name": "ce/main"}]}},
            "Loss log",
            id="a loss name a report could not carry",
        ),
        pytest.param(
            "tasks",
            {"t": {"kind": "classification", "metrics": {"f1/macro": {"name": "f1"}}}},
            "Metric",
            id="a metric label a report could not carry",
        ),
        pytest.param(
            "run",
            {"checkpoint_path": "runs/nothing-here.ckpt"},
            "names no file",
            id="a checkpoint that is not there",
        ),
        pytest.param(
            "teacher",
            {"name": "timm", "checkpoint_path": "runs/nothing-here.ckpt"},
            "names no file",
            id="a teacher pointed at nothing",
        ),
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


def adapting(module: str) -> dict[str, Any]:
    return {"name": "lora", "module": module, "target_modules": ["qkv"]}


def freezing(*modules: str) -> list[dict[str, Any]]:
    return [{"name": "freeze", "modules": list(modules)}]


@pytest.mark.parametrize(
    ("adapted", "frozen"),
    [
        pytest.param("backbone", ["backbone"], id="the same module"),
        pytest.param("backbone", ["heads.species", "backbone.layer1"], id="a part of what is adapted"),
        pytest.param("backbone.layer1", ["backbone"], id="a module that holds what is adapted"),
    ],
)
def test_freezing_what_a_run_adapts_is_refused_where_both_are_declared(
    minimal: dict[str, Any], adapted: str, frozen: list[str]
) -> None:
    """Held still, a delta never moves, and the run trains its heads alone while the declaration says otherwise."""
    with pytest.raises(ValidationError, match="held still"):
        load_config({**minimal, "adapter": adapting(adapted), "callbacks": freezing(*frozen)})


def teaching(tmp_path: Path) -> dict[str, Any]:
    """A second network to learn from, and the file its weights come out of."""
    kept = tmp_path / "teacher.ckpt"
    kept.write_bytes(b"")
    return {"name": "timm", "model_name": "resnet50", "checkpoint_path": str(kept)}


def test_a_teacher_nobody_would_learn_from_is_refused(minimal: dict[str, Any], tmp_path: Path) -> None:
    """A second network is built, loaded and carried through the run; one nothing reads is pure cost."""
    with pytest.raises(ValidationError, match="learns from the data alone"):
        load_config({**minimal, "teacher": teaching(tmp_path)})


def test_distilling_with_nobody_to_learn_from_is_refused(minimal: dict[str, Any]) -> None:
    """Left alone it dies where the learner is built, naming a fact rather than the section to write."""
    with pytest.raises(ValidationError, match="nothing to distil from"):
        load_config({**minimal, "learner": {"name": "distillation"}})


def test_a_teacher_and_the_learner_that_reads_one_are_declared_together(
    minimal: dict[str, Any], tmp_path: Path
) -> None:
    config = load_config({**minimal, "learner": {"name": "distillation"}, "teacher": teaching(tmp_path)})

    assert config.teacher is not None and config.teacher.spelled == "timm"


def test_a_learner_written_as_an_import_path_keeps_the_teacher_it_was_given(
    minimal: dict[str, Any], tmp_path: Path
) -> None:
    """A `_target_` declaration writes no name to recognise, so the pairing holds for nobody, accusing nobody.

    The limitation the rate monitor and the freeze-against-adapters pairing already accept: read by name,
    a component that declares none is left alone. Refusing it would call a correct run wrong.
    """
    config = load_config(
        {**minimal, "learner": {"_target_": "src.training.DistillationLearner"}, "teacher": teaching(tmp_path)}
    )

    assert config.teacher is not None


def test_freezing_a_part_the_adapter_does_not_reach_is_left_alone(minimal: dict[str, Any]) -> None:
    """The pairing is refused where it would hold the delta still, and nowhere else."""
    config = load_config({**minimal, "adapter": adapting("backbone"), "callbacks": freezing("heads.species")})

    assert config.adapter is not None and config.adapter.params["module"] == "backbone"


class TestTask:
    @pytest.fixture
    def task(self) -> dict[str, Any]:
        return {"kind": "classification", "target_column": "species", "classes": CLASSES}

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

    def test_a_task_declares_the_head_a_run_builds_for_it(self, task: dict[str, Any]) -> None:
        """A task is served by a head over one stream; its output is found under the task's own name."""
        declared = TaskConfig.model_validate({**task, "head": {"name": "linear", "stream": "pooled"}})

        assert declared.head is not None and declared.head.stream == "pooled"

    @pytest.mark.parametrize("inputs", ["", " ", "pooled "], ids=["empty", "blank", "padded"])
    def test_a_head_names_one_feature_stream(self, task: dict[str, Any], inputs: str) -> None:
        with pytest.raises(ValidationError):
            TaskConfig.model_validate({**task, "head": {"name": "linear", "stream": inputs}})

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
