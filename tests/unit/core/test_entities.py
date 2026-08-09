"""Data-entity contracts: Sample/Batch, Features, Prediction, Task, DataProfile."""

from __future__ import annotations

import pytest
import torch

from src.core import (
    AdaptedTarget,
    Batch,
    DataProfile,
    Features,
    Instances,
    Prediction,
    Sample,
    TargetFacts,
)
from tests.support.entities import a_task
from tests.support.narrowing import tensor


def test_sample_holds_loose_values_before_collation() -> None:
    sample = Sample(inputs={"image": [[0.0]]}, targets={"label": 3})

    assert sample.targets["label"] == 3
    assert sample.meta == {}


def test_batch_to_returns_a_new_batch_on_the_device() -> None:
    batch = Batch(
        inputs={"image": torch.zeros(2, 3)},
        targets={"label": torch.tensor([0, 1])},
        meta={"paths": ["a.jpg", "b.jpg"]},
    )

    moved = batch.to("cpu")

    assert moved is not batch
    assert isinstance(moved, Batch)
    assert moved.inputs["image"].device.type == "cpu"
    assert torch.equal(tensor(moved.targets["label"]), tensor(batch.targets["label"]))
    assert moved.meta == {"paths": ["a.jpg", "b.jpg"]}


def test_features_behaves_like_a_read_only_mapping_of_streams() -> None:
    features = Features(streams={"features": torch.zeros(2, 8)})

    assert "features" in features
    assert list(features.keys()) == ["features"]
    assert features["features"].shape == (2, 8)


def test_features_names_available_streams_in_the_error() -> None:
    features = Features(streams={"encoder": torch.zeros(1)})

    with pytest.raises(KeyError, match="encoder"):
        _ = features["decoder"]


def test_prediction_maps_task_names_to_outputs() -> None:
    prediction = Prediction(outputs={"label": torch.zeros(2, 10)})

    assert tensor(prediction.outputs["label"]).shape == (2, 10)
    assert prediction.features is None


def test_a_prediction_carries_no_logits_unless_a_model_offers_them() -> None:
    """Optional and last, so a family with nothing pre-activation to show is unchanged."""
    assert Prediction(outputs={"label": torch.zeros(1)}).logits is None


def test_absent_target_has_empty_views() -> None:
    absent = AdaptedTarget.absent()

    assert absent.for_loss.numel() == 0
    assert absent.for_metrics.numel() == 0


def test_adapted_target_separates_loss_and_metric_views() -> None:
    soft = torch.tensor([[0.9, 0.1]])
    adapted = AdaptedTarget(for_loss=soft, for_metrics=soft.argmax(dim=1))

    assert adapted.for_loss.shape == (1, 2)
    assert adapted.for_metrics.item() == 0


def test_task_defaults_weight_to_one() -> None:
    assert a_task().weight == 1.0


def test_task_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        a_task(weight=0.0)


def test_task_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="name"):
        a_task(name="  ")


def test_task_is_immutable() -> None:
    task = a_task()

    with pytest.raises(AttributeError):
        task.weight = 2.0  # type: ignore[misc]


def test_data_profile_stores_facts_inferred_from_data() -> None:
    profile = DataProfile()
    profile.record("label", TargetFacts(num_classes=10))

    assert profile.require_num_classes("label") == 10


def test_data_profile_explains_missing_facts() -> None:
    profile = DataProfile()

    with pytest.raises(LookupError, match="label"):
        profile.require_num_classes("label")


def test_ground_truth_instances_carry_no_scores() -> None:
    """One entity serves both sides of a comparison, so a metric needs no second signature.

    The asymmetry that is real — annotation has no confidence — is stated rather than
    filled with ones, which would read as certainty nobody measured.
    """
    truth = Instances(
        boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        labels=torch.tensor([1]),
        sample_index=torch.tensor([0]),
    )

    assert truth.scores is None


def test_instances_hand_back_one_images_share() -> None:
    """Flat is the only shape a ragged quantity has that a tensor can carry.

    So the split back out belongs with the entity: left to consumers, each writes the
    mask itself and one of them gets it wrong.
    """
    two = Instances(
        boxes=torch.tensor([[0.0, 0, 1, 1], [2.0, 2, 3, 3], [4.0, 4, 5, 5]]),
        labels=torch.tensor([0, 1, 0]),
        sample_index=torch.tensor([0, 1, 1]),
        scores=torch.tensor([0.9, 0.8, 0.7]),
    )

    second = two.of(1)

    assert second.labels.tolist() == [1, 0]
    assert second.boxes.tolist() == [[2.0, 2, 3, 3], [4.0, 4, 5, 5]]
    assert second.scores is not None
    assert second.scores.tolist() == pytest.approx([0.8, 0.7])


def test_instances_refuse_columns_of_different_lengths() -> None:
    """A box without its label is not a partial answer but a silently wrong one.

    The metric would pair box i with label i and report a number nobody can trace back
    to the mismatch that produced it.
    """
    with pytest.raises(ValueError, match="3 boxes"):
        Instances(
            boxes=torch.zeros(3, 4),
            labels=torch.zeros(2, dtype=torch.int64),
            sample_index=torch.zeros(3, dtype=torch.int64),
        )
