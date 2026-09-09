"""The head override: config names the kind, sizes stay derived — scenario ArcFace-classifier."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.core import Batch
from src.losses import ArcFaceCriterion
from src.models import CompositeModel, CosineHead, LinearHead
from tests.support.configs import paper_config, tasks_of
from tests.support.entities import dataset_facts
from tests.support.fakes import FlattenBackbone
from tests.support.narrowing import tensor

FEATURES = 12
PERSON = {"kind": "classification", "target": "person_id", "classes": {0: "0", 1: "1", 2: "2"}}


def experiment(**task_extras: Any) -> Any:
    """The base experiment with one identity task, carrying whatever the test declares on it."""
    return paper_config(tasks={"person": PERSON | task_extras})


FACTS = dataset_facts(person=3)
"""What setup would have learned about the identity task: three people."""


def test_the_declared_kind_arrives_at_the_derived_sizes() -> None:
    """`{name: cosine}` is a complete declaration: no width or class count in config."""
    config = experiment(head={"name": "cosine"})

    _, components = tasks_of(config, FACTS, FlattenBackbone(dim=FEATURES))

    head = components["person"].head
    assert isinstance(head, CosineHead)
    assert head.prototypes.shape == (3, FEATURES)


def test_no_declaration_keeps_the_topology_default() -> None:
    _, components = tasks_of(experiment(), FACTS, FlattenBackbone(dim=FEATURES))

    assert isinstance(components["person"].head, LinearHead)


def test_the_arcface_classifier_trains_end_to_end() -> None:
    """Scenario two whole: cosine head owns the prototypes, arcface is only the training margin."""
    torch.manual_seed(0)
    config = experiment(head={"name": "cosine"}, loss={"name": "arcface", "margin": 0.2})

    _, components = tasks_of(config, FACTS, FlattenBackbone(dim=FEATURES))
    assert isinstance(components["person"].criterion, ArcFaceCriterion)

    model = CompositeModel(backbone=FlattenBackbone(dim=FEATURES), components=components)
    batch = Batch(inputs={"image": torch.randn(4, 3, 2, 2)}, targets={"person_id": torch.tensor([0, 1, 2, 0])})
    result = model.step(Batch(inputs=batch.inputs, targets={"person": batch.targets["person_id"]}))
    result.loss.total.backward()

    prototypes = next(m for m in model.modules() if isinstance(m, CosineHead)).prototypes
    assert torch.isfinite(result.loss.total)
    assert prototypes.grad is not None and bool(prototypes.grad.abs().sum() > 0)
    assert tensor(result.prediction.outputs["person"]).shape == (4, 3)  # classes at inference, no margin


def test_the_native_head_takes_no_arguments() -> None:
    """``native`` names what the backbone brings, built by the backbone; an argument would have nowhere to go."""
    config = experiment(head={"name": "native", "rank": 2})

    with pytest.raises(ValueError, match="'native' takes no arguments"):
        tasks_of(config, FACTS, FlattenBackbone(dim=FEATURES))


def test_the_bare_spelling_of_native_is_the_same_declaration() -> None:
    """``head: native`` is the component grammar's own sugar for ``head: {name: native}``."""
    config = experiment(head="native")

    assert config.tasks["person"].head is not None
    assert config.tasks["person"].head.name == "native"


def test_a_size_written_on_the_head_declaration_is_refused_by_name() -> None:
    """The sizes come from the backbone and the data; a copy in config could only disagree."""
    config = experiment(head={"name": "cosine", "in_features": 5})

    with pytest.raises(ValueError, match="in_features"):
        tasks_of(config, FACTS, FlattenBackbone(dim=FEATURES))


def test_the_components_weight_is_the_tasks_weight_when_the_snapshot_is_taken() -> None:
    """``TaskComponents.weight`` is a compiled snapshot of ``Task.weight`` — the model reads it every step
    and never sees the task — so the two are one number at the moment the kind compiles them."""
    declared = {"label": {"kind": "classification", "target": "label", "classes": {0: "a", 1: "b"}, "weight": 0.5}}

    tasks, components = tasks_of(paper_config(tasks=declared), dataset_facts(), FlattenBackbone(dim=FEATURES))

    assert components["label"].weight == tasks[0].weight == 0.5
