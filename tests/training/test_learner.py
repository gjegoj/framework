"""A step: every task's loss under its own name, the predictions to score, and the targets to score them by."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn

from src.core import Batch, LossOutput, ModelOutput, TargetInfo, TensorTree, require_tensor
from src.losses import Loss
from src.losses.build import build_loss
from src.models import Model
from src.tasks import Classification, Regression, Task
from src.training import StandardLearner

CLASSES = {0: "cat", 1: "dog", 2: "bird"}
LOGITS = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
YEARS = torch.tensor([[1.0], [3.0]])
ENTROPY = "species/cross_entropy"


def number(value: Tensor) -> float:
    """A reported value as a number; detached, because a loss still carries the graph it was built in."""
    return float(value.detach())


class Echo(Model):
    """A network that scales what it was handed: one parameter, so the graph a step builds is visible."""

    def __init__(self, outputs: Mapping[str, Tensor]) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.answers = dict(outputs)

    def forward(self, inputs: Mapping[str, TensorTree]) -> ModelOutput:
        return ModelOutput(outputs={name: value * self.scale for name, value in self.answers.items()})


def tasks(**weights: float) -> dict[str, Task]:
    return {
        "species": Classification("species", TargetInfo(classes=CLASSES), weight=weights.get("species", 1.0)),
        "age": Regression("age", TargetInfo(), weight=weights.get("age", 1.0)),
    }


def losses_for(declared: Mapping[str, Task]) -> dict[str, Loss]:
    return {name: build_loss(task.default_loss, task.info) for name, task in declared.items()}


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


BATCH = batch(torch.tensor([0, 1]))


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

    def test_the_target_it_answers_with_is_the_one_a_metric_scores(self) -> None:
        """MixUp leaves a share of each class in the batch; a metric ranks against the class it mostly is."""
        mixed = batch(torch.tensor([[0.3, 0.7, 0.0], [0.9, 0.1, 0.0]]))

        step = learner().step(mixed)

        assert require_tensor(step.targets["species"], name="species").tolist() == [1, 0]


class Prototypes(Loss):
    """A loss carrying parameters of its own, as an angular margin carries its class prototypes."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.zeros(classes))

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        return self.reported((outputs * self.prototypes).sum())


def test_a_loss_that_carries_parameters_trains_with_the_run_and_stays_out_of_the_model() -> None:
    """ArcFace and its family keep the class prototypes in the loss: they belong to the run, not to the network."""
    declared = tasks()
    owner = StandardLearner(
        Echo({"species": LOGITS, "age": YEARS}),
        declared,
        {"species": Prototypes(len(CLASSES)), "age": losses_for(declared)["age"]},
    )

    step = owner.step(BATCH)
    assert step.loss is not None
    step.loss.total.backward()

    trained = dict(owner.named_parameters())["losses.species.prototypes"]
    assert trained.grad is not None and torch.any(trained.grad != 0)
    assert not any("prototypes" in key for key in owner.model.state_dict())


def test_the_model_and_the_losses_keep_the_paths_a_checkpoint_addresses() -> None:
    """`model.…` is what a freeze callback and a checkpoint name; a loss with parameters of its own rides along."""
    registered = dict(learner().named_modules())

    assert "model" in registered and "losses.species" in registered
