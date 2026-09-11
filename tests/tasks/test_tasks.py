"""A task is what a run learns for one target: the class is the semantics, the instance is this run's facts."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.core import Batch, ModelOutput, TargetInfo, require_tensor
from src.tasks import Task
from src.tasks.registry import task_registry
from tests.support.tasks import info, specimen


def batch(target: Tensor) -> Batch:
    return Batch(inputs={}, targets={"t": target}, count=len(target))


def prediction(logits: Tensor, name: str = "t") -> ModelOutput:
    return ModelOutput(outputs={name: logits})


class TestContract:
    @pytest.mark.parametrize("name", list(task_registry))
    def test_every_registered_task_declares_what_a_run_needs_to_assemble_it(self, name: str) -> None:
        """A kind a config may name has to answer these without a run: they choose encoder, head and metrics."""
        kind = task_registry.get(name)

        assert issubclass(kind, Task)
        assert isinstance(kind.default_head["input"], str) and isinstance(kind.default_head["name"], str)
        assert kind.default_target_encoder is None or isinstance(kind.default_target_encoder, str)
        assert isinstance(kind.default_metrics, dict)

    @pytest.mark.parametrize("kind", list(task_registry))
    def test_every_registered_task_answers_the_step_with_tensors(self, kind: str) -> None:
        """Whatever the semantics, one step is: a target for the loss, a prediction, and a target to score it."""
        task, output, step = specimen(kind)

        assert isinstance(task.loss_target(step), Tensor)
        assert isinstance(require_tensor(task.postprocess(output), name=task.name), Tensor)
        assert isinstance(task.metric_view(step), Tensor)

    def test_a_task_carries_its_name_its_facts_and_its_weight(self) -> None:
        task = task_registry.get("classification")("species", info(), weight=0.5)

        assert task.name == "species" and task.info.num_classes == 3 and task.weight == 0.5

    @pytest.mark.parametrize("weight", [0.0, -1.0, float("nan")], ids=["zero", "negative", "not a number"])
    def test_a_weight_that_cannot_scale_a_loss_is_refused(self, weight: float) -> None:
        with pytest.raises(ValueError, match="weight"):
            task_registry.get("classification")("t", info(), weight=weight)

    def test_a_target_the_batch_does_not_carry_is_named(self) -> None:
        task = task_registry.get("classification")("species", info())

        with pytest.raises(LookupError, match="species"):
            task.loss_target(Batch(inputs={}, count=2))


class TestClassification:
    @pytest.mark.parametrize(
        ("kind", "declared", "shape"),
        [
            pytest.param("classification", info(), (3,), id="one class per sample"),
            pytest.param("binary_classification", TargetInfo(), (1,), id="one score per sample"),
            pytest.param("multilabel_classification", info(), (3,), id="a score per label"),
            pytest.param("segmentation", info(), (3, None, None), id="one class per pixel"),
            pytest.param("binary_segmentation", TargetInfo(), (1, None, None), id="one score per pixel"),
        ],
    )
    def test_the_output_a_head_must_produce_follows_from_the_semantics_and_the_vocabulary(
        self, kind: str, declared: TargetInfo, shape: tuple[int | None, ...]
    ) -> None:
        assert task_registry.get(kind).output_shape(declared).sizes == shape

    @pytest.mark.parametrize("kind", ["classification", "segmentation"])
    def test_a_vocabulary_too_small_to_choose_from_is_refused(self, kind: str) -> None:
        with pytest.raises(ValueError, match="two"):
            task_registry.get(kind).output_shape(TargetInfo(classes={0: "only"}))

    def test_probabilities_come_back_over_the_class_axis(self) -> None:
        task = task_registry.get("classification")("t", info())

        probabilities = require_tensor(task.postprocess(prediction(torch.rand(2, 3))), name="t")

        assert probabilities.shape == (2, 3) and torch.allclose(probabilities.sum(-1), torch.ones(2))

    def test_a_dense_task_keeps_the_map_and_normalizes_each_pixel(self) -> None:
        task = task_registry.get("segmentation")("t", info())

        probabilities = require_tensor(task.postprocess(prediction(torch.rand(2, 3, 4, 5))), name="t")

        assert probabilities.shape == (2, 3, 4, 5) and torch.allclose(probabilities.sum(1), torch.ones(2, 4, 5))

    def test_a_binary_task_scores_between_zero_and_one(self) -> None:
        task = task_registry.get("binary_classification")("t", TargetInfo())

        scores = require_tensor(task.postprocess(prediction(torch.randn(4, 1))), name="t")

        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_a_mixed_target_stays_soft_for_the_loss_and_hardens_for_metrics(self) -> None:
        """MixUp trains against a share of each class; a metric ranks against the class the sample mostly is."""
        task = task_registry.get("classification")("t", info())
        mixed = batch(torch.tensor([[0.3, 0.7, 0.0], [0.9, 0.1, 0.0]]))

        assert task.loss_target(mixed).dtype.is_floating_point
        assert task.metric_view(mixed).tolist() == [1, 0]

    def test_a_plain_class_index_reaches_the_loss_as_an_index(self) -> None:
        task = task_registry.get("classification")("t", info())

        assert task.loss_target(batch(torch.tensor([2, 0]))).dtype is torch.long


class TestRegression:
    def test_a_plain_target_is_one_number_per_sample(self) -> None:
        task = task_registry.get("regression")("t", TargetInfo())

        prepared = require_tensor(task.postprocess(prediction(torch.tensor([[1.5], [2.5]]))), name="t")

        assert task.output_shape(TargetInfo()).sizes == (1,) and prepared.tolist() == [1.5, 2.5]

    def test_a_binned_target_is_learned_as_a_distribution_and_read_back_as_the_value_it_stands_for(self) -> None:
        """The same number, two views: bins for the loss, the value they average to for a metric."""
        binned = TargetInfo(classes={0: "0", 1: "1", 2: "2"}, values=(0.0, 10.0, 20.0))
        task = task_registry.get("regression")("t", binned)
        distribution = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])

        value = require_tensor(task.postprocess(prediction(distribution.log())), name="t")

        assert task.output_shape(binned).sizes == (3,)
        assert value.tolist() == pytest.approx([10.0, 5.0], abs=0.1)
        assert task.metric_view(batch(distribution)).tolist() == pytest.approx([10.0, 5.0])

    def test_its_default_loss_follows_the_target_encoder_that_laid_out_the_bins(self) -> None:
        plain = task_registry.get("regression")("t", TargetInfo())
        binned = task_registry.get("regression")("t", TargetInfo(classes={0: "a", 1: "b"}, values=(0.0, 1.0)))

        assert plain.default_loss == "mse"
        assert isinstance(binned.default_loss, list) and len(binned.default_loss) == 2
