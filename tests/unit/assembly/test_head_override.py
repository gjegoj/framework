"""The head override: config names the kind, sizes stay derived — scenario ArcFace-classifier."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.assembly.tasks import build_tasks
from src.core import Batch, DataProfile, TargetFacts
from src.losses import ArcFaceCriterion
from src.models import CompositeModel, CosineHead, LinearHead
from tests.support.configs import paper_config
from tests.support.fakes import FlattenBackbone
from tests.support.narrowing import tensor

FEATURES = 12
PERSON = {"preset": "classification", "target": "person_id"}


def experiment(**task_extras: Any) -> Any:
    """The base experiment with one identity task, carrying whatever the test declares on it."""
    return paper_config(tasks={"person": PERSON | task_extras})


def profile() -> DataProfile:
    built = DataProfile()
    built.record("person", TargetFacts(num_classes=3))
    return built


def test_the_declared_kind_arrives_at_the_derived_sizes() -> None:
    """`{name: cosine}` is a complete declaration: no width or class count in config."""
    config = experiment(head={"name": "cosine"})

    _, components = build_tasks(config, profile(), FlattenBackbone(dim=FEATURES))

    head = components["person"].head
    assert isinstance(head, CosineHead)
    assert head.prototypes.shape == (3, FEATURES)


def test_no_declaration_keeps_the_topology_default() -> None:
    _, components = build_tasks(experiment(), profile(), FlattenBackbone(dim=FEATURES))

    assert isinstance(components["person"].head, LinearHead)


def test_the_arcface_classifier_trains_end_to_end() -> None:
    """Scenario two whole: cosine head owns the prototypes, arcface is only the training margin."""
    torch.manual_seed(0)
    config = experiment(head={"name": "cosine"}, loss={"name": "arcface", "margin": 0.2})

    _, components = build_tasks(config, profile(), FlattenBackbone(dim=FEATURES))
    assert isinstance(components["person"].criterion, ArcFaceCriterion)

    model = CompositeModel(backbone=FlattenBackbone(dim=FEATURES), components=components)
    batch = Batch(inputs={"image": torch.randn(4, 3, 2, 2)}, targets={"person_id": torch.tensor([0, 1, 2, 0])})
    result = model.step(Batch(inputs=batch.inputs, targets={"person": batch.targets["person_id"]}))
    result.loss.total.backward()

    prototypes = next(m for m in model.modules() if isinstance(m, CosineHead)).prototypes
    assert torch.isfinite(result.loss.total)
    assert prototypes.grad is not None and bool(prototypes.grad.abs().sum() > 0)
    assert tensor(result.prediction.outputs["person"]).shape == (4, 3)  # classes at inference, no margin


def test_a_head_beside_native_head_is_refused() -> None:
    with pytest.raises(ValueError, match="not both"):
        experiment(head={"name": "cosine"}, native_head=True)
