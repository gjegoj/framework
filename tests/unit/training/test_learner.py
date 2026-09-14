"""A step: every task's loss under its own name, the predictions to score, and the targets to score them by."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from src.core import Batch, LossOutput, TargetInfo, require_tensor
from src.losses import Loss
from src.losses.build import build_loss
from src.tasks import Classification, Regression, Task
from src.training import StandardLearner
from tests.support.losses import LearnedMargin
from tests.support.models import Angles, Echo

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
INFO = TargetInfo(classes=CLASSES)
LOGITS = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
YEARS = torch.tensor([[1.0], [3.0]])
ENTROPY = "species/cross_entropy"


def number(value: Tensor) -> float:
    """A reported value as a number; detached, because a loss still carries the graph it was built in."""
    return float(value.detach())


def tasks(**weights: float) -> dict[str, Task]:
    return {
        "species": Classification("species", INFO, weight=weights.get("species", 1.0)),
        "age": Regression("age", TargetInfo(), weight=weights.get("age", 1.0)),
    }


def losses_for(declared: Mapping[str, Task]) -> dict[str, Loss]:
    return {name: build_loss(task.default_loss, task.facts()) for name, task in declared.items()}


def learner(**weights: float) -> StandardLearner:
    """Two tasks over one network, each judged by the loss its own semantics implies."""
    declared = tasks(**weights)
    return StandardLearner(Echo({"species": LOGITS, "age": YEARS}), declared, losses_for(declared))


def batch(species: Tensor) -> Batch:
    return Batch(
        inputs={"image": torch.zeros(2, 3, 4, 4)},
        targets={"species": species, "age": torch.tensor([1.0, 2.0])},
        count=2,
    )


SPECIES = torch.tensor([0, 1])
BATCH = batch(SPECIES)


class TestObjective:
    def test_every_tasks_loss_is_reported_under_its_own_name(self) -> None:
        step = learner().step(BATCH)

        assert step.loss is not None and set(step.loss.losses) == {ENTROPY, "age/mse"}

    def test_a_weight_scales_what_a_task_contributes_and_not_what_it_reports(self) -> None:
        """A chart of the raw losses stays comparable between runs; only the objective moves with the weight."""
        plain, halved = learner().step(BATCH), learner(species=0.5).step(BATCH)
        assert plain.loss is not None and halved.loss is not None

        assert number(halved.loss.losses[ENTROPY]) == number(plain.loss.losses[ENTROPY])
        assert number(halved.loss.contributions[ENTROPY]) == pytest.approx(
            number(plain.loss.contributions[ENTROPY]) * 0.5
        )

    def test_the_total_is_what_the_tasks_contribute_together(self) -> None:
        step = learner(species=0.5, age=2.0).step(BATCH)
        assert step.loss is not None

        contributed = sum(number(value) for value in step.loss.contributions.values())
        assert number(step.loss.total) == pytest.approx(contributed)

    def test_a_task_the_model_does_not_answer_for_is_named(self) -> None:
        declared = tasks()
        incomplete = StandardLearner(Echo({"species": LOGITS}), declared, losses_for(declared))

        with pytest.raises(LookupError, match="age"):
            incomplete.step(BATCH)

    def test_a_task_with_nothing_to_learn_it_by_is_refused_where_the_learner_is_assembled(self) -> None:
        declared = tasks()

        with pytest.raises(ValueError, match="age"):
            StandardLearner(Echo({}), declared, {"species": losses_for(declared)["species"]})


class TestViews:
    def test_a_prediction_leaves_the_graph_behind_and_the_loss_keeps_it(self) -> None:
        """Metrics and displays never go backward; a prediction built inside the graph would hold it alive."""
        step = learner().step(BATCH)
        assert step.loss is not None

        assert step.loss.total.requires_grad
        assert not require_tensor(step.predictions["species"], name="species").requires_grad

    def test_a_prediction_is_what_the_task_says_its_output_means(self) -> None:
        probabilities = require_tensor(learner().step(BATCH).predictions["species"], name="species")

        assert probabilities.shape == (2, 3) and torch.allclose(probabilities.sum(-1), torch.ones(2))

    def test_a_prediction_is_what_the_artifact_would_ship_rather_than_a_reading_of_the_steps_own(self) -> None:
        """A step and an artifact publish one task's numbers through one rule, so they cannot disagree.

        The page draws what a step answered with. A step that read a network's angles as though they
        were a projection would put a chip reading 7% under a picture the model is certain about —
        measured at 37 classes it could never read above 17% — while the artifact shipped beside it
        publishes the angle itself, and the two would be describing different models.
        """
        task = Classification("species", INFO)
        network = Angles(task.name, reads="image", in_features=6, out_features=task.out_features())
        step = StandardLearner(network, {task.name: task}, {task.name: build_loss("arcface", task.facts())})
        features = torch.randn(2, 6)

        answered = step.step(Batch(inputs={"image": features}, targets={"species": SPECIES}, count=2))

        assert torch.allclose(require_tensor(answered.predictions[task.name], name=task.name), network.head(features))

    def test_a_task_whose_identities_evaluation_does_not_share_is_not_scored_there(self) -> None:
        """Its objective keeps one prototype per training identity, and evaluation names others.

        Not a preference — arithmetic. Measured live on the shipped example: the encoder learned 26
        identities from the training split, and the first validation batch stopped the run with
        "Class values must be smaller than num_classes". There is no number to report here, so none is.
        """
        declared = tasks()
        step = StandardLearner(
            Echo({"species": LOGITS, "age": YEARS}), declared, losses_for(declared), learned_only=["species"]
        ).eval()

        answered = step.step(BATCH)

        assert answered.loss is not None
        assert set(answered.loss.breakdown()) == {"age/mse"}
        assert set(answered.predictions) == {"species", "age"}, "every task still answers what a metric scores"

    def test_the_same_task_is_scored_while_the_run_is_learning_it(self) -> None:
        """Training is where those prototypes are learned, so training is where the number means something."""
        declared = tasks()
        learning = StandardLearner(
            Echo({"species": LOGITS, "age": YEARS}), declared, losses_for(declared), learned_only=["species"]
        ).train()

        assert ENTROPY in (learning.step(BATCH).loss or LossOutput(torch.zeros(()))).breakdown()

    def test_a_step_left_with_no_objective_at_all_answers_with_no_loss_rather_than_a_zero(self) -> None:
        """What every run of this kind does outside training: `val/loss` is absent, not nought."""
        declared: dict[str, Task] = {"species": Classification("species", INFO)}
        objective = {"species": build_loss("cross_entropy", declared["species"].facts())}
        alone = StandardLearner(Echo({"species": LOGITS}), declared, objective, learned_only=["species"]).eval()

        assert alone.step(batch(torch.tensor([0, 1]))).loss is None

    def test_a_learner_told_of_a_task_it_does_not_learn_says_so(self) -> None:
        declared = tasks()
        with pytest.raises(ValueError, match="breed"):
            StandardLearner(
                Echo({"species": LOGITS, "age": YEARS}), declared, losses_for(declared), learned_only=["breed"]
            )

    def test_the_target_it_answers_with_is_the_one_a_metric_scores(self) -> None:
        """MixUp leaves a share of each class in the batch; a metric ranks against the class it mostly is."""
        mixed = batch(torch.tensor([[0.3, 0.7, 0.0], [0.9, 0.1, 0.0]]))

        step = learner().step(mixed)

        assert require_tensor(step.targets["species"], name="species").tolist() == [1, 0]


def test_a_loss_that_carries_parameters_trains_with_the_run_and_stays_out_of_the_model() -> None:
    """ArcFace and its family keep the class prototypes in the loss: they belong to the run, not to the network."""
    declared = tasks()
    owner = StandardLearner(
        Echo({"species": LOGITS, "age": YEARS}),
        declared,
        {"species": LearnedMargin(len(CLASSES)), "age": losses_for(declared)["age"]},
    )

    step = owner.step(BATCH)
    assert step.loss is not None
    step.loss.total.backward()

    trained = dict(owner.named_parameters())["losses.species.margin"]
    assert trained.grad is not None and torch.any(trained.grad != 0)
    assert not any("margin" in key for key in owner.model.state_dict())


class TestParameterGroups:
    def test_a_task_with_parts_of_its_own_becomes_a_group_named_after_it(self) -> None:
        """The groups are also what a learning-rate chart draws one line each of, so they are named."""
        assert {group["name"] for group in learner().parameter_groups()} == {"backbone", "species", "age"}

    def test_what_no_task_claims_is_one_shared_group(self) -> None:
        built = learner()
        shared = next(group for group in built.parameter_groups() if group["name"] == "backbone")

        assert [id(parameter) for parameter in shared["params"]] == [id(built.model.shared)]

    def test_every_parameter_is_claimed_exactly_once(self) -> None:
        """A parameter in two groups is optimized twice; one in none is never trained at all."""
        built = learner()

        claimed = [id(parameter) for group in built.parameter_groups() for parameter in group["params"]]

        assert sorted(claimed) == sorted(id(parameter) for parameter in built.parameters())

    def test_a_loss_that_owns_parameters_is_grouped_with_its_task(self) -> None:
        """An angular margin keeps class prototypes: they follow the task's rate, not the encoder's."""
        declared = tasks()
        owner = StandardLearner(
            Echo({"species": LOGITS, "age": YEARS}),
            declared,
            {"species": LearnedMargin(len(CLASSES)), "age": losses_for(declared)["age"]},
        )

        grouped = {group["name"]: {id(one) for one in group["params"]} for group in owner.parameter_groups()}

        assert id(dict(owner.named_parameters())["losses.species.margin"]) in grouped["species"]

    def test_a_task_that_declares_a_rate_carries_it_on_its_group(self) -> None:
        rated = Classification("species", INFO, lr=1e-2)
        built = StandardLearner(
            Echo({"species": LOGITS}), {"species": rated}, {"species": build_loss("cross_entropy", rated.facts())}
        )

        assert {group["name"]: group.get("lr") for group in built.parameter_groups()} == {
            "backbone": None,
            "species": 1e-2,
        }

    def test_a_task_with_nothing_of_its_own_cannot_declare_a_rate(self) -> None:
        """A rate over no parameters is a declaration that silently does nothing."""
        declared = {"age": Regression("age", TargetInfo(), lr=1e-2)}
        built = StandardLearner(Echo({}), declared, {"age": build_loss("mse", Regression("t", TargetInfo()).facts())})

        with pytest.raises(ValueError, match="age"):
            built.parameter_groups()


def test_the_model_and_the_losses_keep_the_paths_a_checkpoint_addresses() -> None:
    """`model.…` is what a freeze callback and a checkpoint name; a loss with parameters of its own rides along."""
    registered = dict(learner().named_modules())

    assert "model" in registered and "losses.species" in registered
