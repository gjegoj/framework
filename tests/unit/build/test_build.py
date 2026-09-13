"""The composition root: a validated declaration becomes a run, in the one order the contracts allow.

What is under test is the wiring rather than the parts — which facts travel from the data to the model,
what a task gets when it declares nothing, which splits a run prepares, and what the trainer is handed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.build import build, metrics_for
from src.config import TaskConfig, load_config
from src.core import Axis, TargetInfo
from src.tasks import Classification, Segmentation
from src.tasks.build import head_for
from tests.support.declarations import NORMALIZATION
from tests.unit.build.conftest import SIZE


def experiment(declaration: Mapping[str, Any], **overrides: Any) -> Any:
    return build(load_config({**declaration, **overrides}))


class TestFactsTravel:
    def test_the_head_is_sized_by_what_the_data_settled_and_not_by_the_declaration(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """The one ordering the root exists to hold: the data is read before the model is built."""
        built = experiment(declaration)

        batch = next(iter(built.data.train_dataloader()))
        outputs = built.module.learner.model(batch.inputs).outputs["species"]
        assert outputs.shape == (2, 2), "two samples, and one column per class the data settled"

    def test_a_size_only_the_fitted_data_knows_is_what_the_model_is_built_to(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """Bins are laid out over the training split's own range, so this width exists only after the fit."""
        binned = {"name": "linear_bins", "bins": 3}
        tasks = {"age": {"kind": "regression", "target_column": "age", "target_encoder": binned}}

        built = experiment(declaration, tasks=tasks)

        batch = next(iter(built.data.train_dataloader()))
        assert built.module.learner.model(batch.inputs).outputs["age"].shape == (2, 3)

    def test_a_task_becomes_an_object_carrying_what_the_run_declared_about_it(
        self, declaration: Mapping[str, Any]
    ) -> None:
        tasks = dict(declaration["tasks"])
        tasks["species"] = {**tasks["species"], "weight": 0.5, "lr": 1.0e-4}

        task = experiment(declaration, tasks=tasks).module.learner.tasks["species"]

        assert isinstance(task, Classification) and (task.weight, task.lr) == (0.5, 1.0e-4)
        assert task.output_shape().size(Axis.CLASSES) == 2

    def test_a_head_holding_the_prototypes_and_the_objective_that_reads_them_fit_with_nothing_written_twice(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """The other metric-learning arrangement, assembled: an ordinary classification task throughout.

        Only the root sees both halves, and the halves are what could disagree — the head has to produce
        cosines, the objective refuses anything else, and the prototype table has to be as wide as the
        space the head projects into. None of those three numbers is written in the declaration twice.
        """
        tasks = dict(declaration["tasks"])
        tasks["species"] = {
            **tasks["species"],
            "head": {"name": "cosine", "stream": "pooled", "embedding_dim": 16},
            "loss": {"name": "arcface", "margin": 0.5},
        }
        built = experiment(declaration, tasks=tasks)

        batch = next(iter(built.data.train_dataloader()))
        step = built.module.learner.step(batch)

        head = built.module.learner.model.heads["species"]
        assert head.prototypes.shape == (2, 16), "one per declared class, as wide as the head projects"
        assert step.loss is not None and bool(step.loss.total.isfinite())
        assert not list(built.module.learner.losses["species"].parameters()), "they are the network's here"

    def test_a_task_that_declares_no_loss_is_learned_by_the_one_its_kind_implies(
        self, declaration: Mapping[str, Any]
    ) -> None:
        built = experiment(declaration)

        assert type(built.module.learner.losses["species"]).__name__ == "CrossEntropy"

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            pytest.param({}, set(Classification.default_metrics), id="the kind's own set"),
            pytest.param({"metrics": {"accuracy": {"name": "accuracy"}}}, {"accuracy"}, id="replaced whole"),
        ],
    )
    def test_a_declared_set_of_metrics_replaces_the_kinds_rather_than_adding_to_it(
        self, declared: Mapping[str, Any], expected: set[str]
    ) -> None:
        task = Classification("species", TargetInfo(classes={0: "cat", 1: "dog"}))
        config = TaskConfig.model_validate({"kind": "classification", "target_column": "species", **declared})

        assert set(metrics_for(config, task)) == expected


class TestWhatTheHeadAnswersWith:
    """One tensor, two separately built readers: only the root sees both, so only the root can refuse.

    A head's numbers are a projection or an angle, and nothing about their shape says which. Every
    objective here would train on either and report a plausible number for the length of a run.
    """

    @pytest.fixture
    def tasks(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return dict(declaration["tasks"])

    def test_an_objective_that_reads_angles_refuses_a_head_that_only_projects(
        self, declaration: Mapping[str, Any], tasks: dict[str, Any]
    ) -> None:
        """`loss: arcface` over the default head adds its margin to something that is not an angle.

        Matched on the ordered pair and not on the word `cosines`, which the message carries whichever
        way round the mismatch is: this test and the one below would otherwise each pass on the other's
        failure, and neither would be watching the direction its name claims.
        """
        tasks["species"] = {**tasks["species"], "loss": {"name": "arcface"}}

        with pytest.raises(ValueError, match="answers with projected, and objective 'arcface' reads cosines"):
            experiment(declaration, tasks=tasks)

    def test_an_ordinary_objective_refuses_a_head_that_answers_with_angles(
        self, declaration: Mapping[str, Any], tasks: dict[str, Any]
    ) -> None:
        """The other way round, which nothing was watching: cross-entropy over values bounded by one.

        Such a run trains and its argmax is even right, while the confidence it reports can never
        leave the neighbourhood of uniform — the scale that would fix that belongs to an objective
        this task did not declare.
        """
        tasks["species"] = {**tasks["species"], "head": {"name": "cosine", "stream": "pooled"}}

        with pytest.raises(ValueError, match="answers with cosines, and objective 'cross_entropy' reads projected"):
            experiment(declaration, tasks=tasks)

    def test_a_head_the_run_wrote_itself_is_taken_at_the_word_it_declares(
        self, declaration: Mapping[str, Any], tasks: dict[str, Any]
    ) -> None:
        """No list of shipped classes anywhere: a head reached by import path says what it answers with."""
        tasks["species"] = {
            **tasks["species"],
            "head": {"_target_": "tests.support.models.OwnCosineHead", "stream": "pooled"},
            "loss": {"name": "arcface"},
        }

        built = experiment(declaration, tasks=tasks)

        step = built.module.learner.step(next(iter(built.data.train_dataloader())))
        assert step.loss is not None and bool(step.loss.total.isfinite())


class TestTwoTasks:
    """What the framework is for, and the only shape where several of its rules meet at once."""

    @pytest.fixture
    def both(self, declaration: Mapping[str, Any]) -> Any:
        tasks = {
            "species": {**declaration["tasks"]["species"], "weight": 0.5},
            "age": {"kind": "regression", "target_column": "age", "lr": 1.0e-4},
        }
        return experiment(declaration, tasks=tasks)

    def test_each_task_is_learned_by_its_own_loss_under_its_own_name(self, both: Any) -> None:
        batch = next(iter(both.data.train_dataloader()))

        step = both.module.learner.step(batch)

        assert step.loss is not None
        assert set(step.loss.losses) == {"species/cross_entropy", "age/mse"}
        assert step.loss.total.ndim == 0

    def test_the_optimizer_holds_a_group_per_task_and_one_for_what_they_share(self, both: Any) -> None:
        groups = both.module.learner.parameter_groups()

        assert [group["name"] for group in groups] == ["backbone", "species", "age"]
        assert "lr" not in groups[1] and groups[2]["lr"] == 1.0e-4

    def test_every_task_is_measured_and_reported_apart(self, both: Any) -> None:
        """One epoch through the trainer the root built: every task keeps its own keys, and its share."""
        both.trainer.fit(both.module, datamodule=both.data)

        logged = set(both.trainer.logged_metrics)
        assert {"train/species/f1/mean", "train/age/mae", "train/loss"} <= logged
        assert "train/species/cross_entropy/contribution" in logged, "a weighted task reports its share"
        assert "train/age/mse/contribution" not in logged, "and an unweighted one has nothing to show twice"


class TestTheHead:
    """A head is two decisions: which kind of head, and which of the backbone's streams it reads."""

    @pytest.fixture
    def task(self) -> Segmentation:
        return Segmentation("mask", TargetInfo(classes={0: "pet", 1: "background"}))

    def test_a_task_that_declares_no_head_gets_the_one_its_kind_serves_itself_with(self, task: Segmentation) -> None:
        declared = TaskConfig.model_validate({"kind": "segmentation", "target_column": "mask_path"})

        assert head_for(declared, task).name == "conv"

    def test_a_declared_head_reads_the_stream_its_kind_reads_unless_it_names_another(self, task: Segmentation) -> None:
        """Which stream a task reads follows from its topology; which head reads it is the run's to choose."""
        own = TaskConfig.model_validate({"kind": "segmentation", "target_column": "mask_path", "head": "native"})
        elsewhere = TaskConfig.model_validate(
            {"kind": "segmentation", "target_column": "mask_path", "head": {"name": "native", "stream": "encoder"}}
        )

        assert (head_for(own, task).name, head_for(own, task).stream) == ("native", "decoder")
        assert head_for(elsewhere, task).stream == "encoder"


class TestTheRunItWillBe:
    def test_only_the_splits_this_run_needs_are_prepared(self, declaration: Mapping[str, Any]) -> None:
        """Preparing a split costs a read and a warm pass; a run that will not use it pays neither."""
        built = experiment(declaration, run={**declaration["run"], "test": False})

        built.data.val_dataloader()
        with pytest.raises(LookupError, match="test"):
            built.data.test_dataloader()

    def test_a_run_that_does_not_train_reads_neither_the_training_split_nor_the_validation_one(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """Evaluating weights it was handed, or only shipping them: reading a split to prepare it for
        nobody is the whole cost of the thing such a run is not doing."""
        built = experiment(declaration, run={**declaration["run"], "train": False})

        built.data.test_dataloader()
        with pytest.raises(LookupError, match="train"):
            built.data.train_dataloader()

    def test_an_encoder_that_would_have_to_learn_its_layout_refuses_a_run_that_never_trains(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """What a run gives up by not reading the training split, said where it is given up rather than
        by a size that silently came from somewhere else. Declaring the range is the way to keep it."""
        binned = {"kind": "regression", "target_column": "age", "target_encoder": {"name": "linear_bins", "bins": 4}}

        with pytest.raises(ValueError, match="low and high"):
            experiment(declaration, tasks={"age": binned}, run={**declaration["run"], "train": False})

    def test_a_run_that_will_test_without_a_split_to_test_on_is_refused_before_it_starts(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """Falling back to the validation rows would report optimistic numbers under an honest name."""
        data = {**declaration["data"], "split": {"train": 0.7, "val": 0.3}}

        with pytest.raises(LookupError, match="test"):
            experiment(declaration, data=data)

    def test_the_loader_knobs_a_run_declares_reach_every_stage(self, declaration: Mapping[str, Any]) -> None:
        built = experiment(declaration, loader={"num_workers": 0, "pin_memory": True})

        assert built.data.val_dataloader().pin_memory is True
        assert built.data.train_dataloader().batch_size == 2


class TestDeclarationsThatCannotHold:
    """Rules about two sections at once: nothing but the root can see both, so nothing else can refuse."""

    def test_a_format_the_framework_never_heard_of_is_refused_before_a_row_is_read(
        self, declaration: Mapping[str, Any], tmp_path: Path
    ) -> None:
        """An export declaration reads nothing but itself, while preparing the data is a source read, an
        encoder fit and a cache warm — minutes to spend before answering a typo in a name.

        Declared here over a source that does not exist, so whichever refusal arrives first is the one a
        run would have paid that time for.
        """
        declared = {
            **declaration,
            "data": {**declaration["data"], "source": str(tmp_path / "no-such-table.csv")},
            "export": [{"name": "onxn"}],
        }

        with pytest.raises(LookupError, match="onxn"):
            build(load_config(declared))

    def test_inputs_bound_to_columns_and_inputs_encoded_are_one_vocabulary(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """Left to itself the mismatch surfaces on the first batch, inside a loader worker.

        Named differently rather than merely declared differently: `data.inputs` and
        `preprocessing.inputs` are two halves of one vocabulary, and it is the names that have to meet.
        """
        encoded = {"name": "image", "image_size": SIZE, **NORMALIZATION}
        preprocessing = {"name": "standard", "inputs": {"picture": encoded}}

        with pytest.raises(ValueError, match=r"data\.inputs has image, preprocessing\.inputs has picture"):
            experiment(declaration, preprocessing=preprocessing)

    def test_a_task_named_after_something_torch_keeps_for_itself_is_refused(
        self, declaration: Mapping[str, Any]
    ) -> None:
        """A task name has to survive two rule sets: this framework's, and the one torch keeps quietly.

        Through the builder rather than the check alone, because the collision happens where the name
        becomes a child of a module — which is what a reader's task name eventually is.
        """
        tasks = {"training": {"kind": "classification", "target_column": "species", "classes": {0: "cat", 1: "dog"}}}

        with pytest.raises(ValueError, match="training"):
            experiment(declaration, tasks=tasks)

    def test_a_head_declared_against_a_model_that_arrives_whole_is_refused(
        self, declaration: Mapping[str, Any]
    ) -> None:
        tasks = {"species": {**declaration["tasks"]["species"], "head": {"name": "linear", "stream": "pooled"}}}

        with pytest.raises(ValueError, match="own heads"):
            experiment(declaration, model={"_target_": "tests.e2e.test_custom_extension.Tiny"}, tasks=tasks)


class TestTheTrainer:
    def test_a_run_that_declares_no_tracker_records_nowhere_at_all(self, declaration: Mapping[str, Any]) -> None:
        """Left to itself Lightning would start a logger of its own, which is not what `tracker: none` says."""
        assert experiment(declaration).trainer.logger is None

    def test_a_declared_tracker_is_what_the_trainer_records_to(
        self, declaration: Mapping[str, Any], tmp_path: Any
    ) -> None:
        built = experiment(declaration, tracker={"name": "csv", "save_dir": str(tmp_path)})

        assert isinstance(built.trainer.logger, CSVLogger)

    def test_the_declared_callbacks_arrive_in_the_order_they_were_written(self, declaration: Mapping[str, Any]) -> None:
        """With a tracker, because one of the two only reports and a run without one is refused by name."""
        built = experiment(
            declaration,
            tracker={"name": "csv", "save_dir": declaration["run"]["directory"]},
            callbacks=[
                {"name": "lr_monitor", "logging_interval": "epoch"},
                {"name": "checkpoint", "monitor": "val/loss", "mode": "min", "save_top_k": 1},
            ],
        )

        declared = [one for one in built.trainer.callbacks if isinstance(one, LearningRateMonitor | ModelCheckpoint)]
        assert [type(one).__name__ for one in declared] == ["LearningRateMonitor", "ModelCheckpoint"]

    def test_the_run_is_as_long_as_the_declaration_says(self, declaration: Mapping[str, Any]) -> None:
        assert experiment(declaration, epochs=3).trainer.max_epochs == 3
