"""Every shipped kind, against the contract a kind signs: from a two-class declaration, its parts agree.

``test_kinds.py`` proves each kind's own decisions one by one; this file proves what every
kind must satisfy together, parametrized over the registry so a kind added tomorrow is
checked without a line written here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
import pytest
import torch
from torch import Tensor

from src.config import MetricConfig
from src.core import Backbone, Batch, Features, Sample, Stage, Stream, TaskFacts
from src.data.collate import collate_samples
from src.data.encoders.base import VocabularyTargetEncoder
from src.data.registry import target_encoder_registry
from src.metrics.build import build_metric_sets
from src.models import CompositeModel
from src.tasks.registry import task_kind_registry
from src.visualization.annotators import DrawingKnobs
from tests.support.entities import a_task

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.tasks import TaskKind
    from src.visualization.entities import SampleView

KINDS = sorted(str(name) for name in task_kind_registry)
TRAINABLE = [name for name in KINDS if task_kind_registry.create(name).unavailable is None]

NOT_YET: dict[str, str] = {}
"""Kinds whose defaults do not agree with each other — none today; a new entry is a measured debt.

``xfail(strict=True)`` so the day a default is fixed is loud: the test passes, the marker
fails, and somebody deletes the entry.
"""

STEPPING = [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=NOT_YET[name])) if name in NOT_YET else name
    for name in TRAINABLE
]

CLASSES = {0: "cat", 1: "dog"}
WIDTH = 8
SIDE = 4

CELLS: Mapping[str, object] = {
    "label": "cat",
    "scalar": 1.0,
    "multilabel": "cat,dog",
    "mask": "mask.png",
}
"""One raw cell per encoder a kind starts from — what a table holds before any encoder sees it."""


class EveryStreamBackbone(Backbone):
    """A pooled stream, a dense one and two stacked views, so every shipped kind finds what it reads."""

    def forward(self, inputs: dict[str, Tensor]) -> Features:
        image = inputs["image"]
        pooled = image.mean(dim=(2, 3)).repeat(1, WIDTH // 3 + 1)[:, :WIDTH]
        dense = image.repeat(1, WIDTH // 3 + 1, 1, 1)[:, :WIDTH]
        views = torch.stack([pooled, pooled.flip(1)], dim=1)
        return Features(streams={Stream.FEATURES: pooled, Stream.DECODER: dense, Stream.EMBEDDINGS: views})

    def feature_dims(self) -> Mapping[str, int]:
        return {Stream.FEATURES: WIDTH, Stream.DECODER: WIDTH, Stream.EMBEDDINGS: WIDTH}


def through_the_kinds_encoder(kind: TaskKind, task_name: str, root: Path) -> tuple[TaskFacts, Batch]:
    """The facts the kind's own encoder reports from two classes, and a batch of the values it produced.

    Sizes come from the data: the facts are read off the fitted encoder as ``setup()`` reads
    them, never written by hand — a kind whose encoder reports no classes is built without.
    """
    encoder_name = kind.default_encoder
    if encoder_name is None:
        samples = [Sample(inputs={"image": torch.rand(3, SIDE, SIDE)}, targets={}) for _ in range(2)]
        return TaskFacts(), collate_samples(samples)
    factory = target_encoder_registry.get(encoder_name)
    arguments: dict[str, object] = {"classes": CLASSES} if issubclass(factory, VocabularyTargetEncoder) else {}  # type: ignore[arg-type]
    if encoder_name == "mask":
        cv2.imwrite(str(root / "mask.png"), np.array([[0, 1] * 2] * SIDE, dtype=np.uint8))
        arguments["root"] = root
    cell = CELLS[encoder_name]
    encoder = target_encoder_registry.create(encoder_name, **arguments).fit([cell, cell])
    facts = TaskFacts(
        num_classes=encoder.num_classes,
        class_names=tuple(names) if (names := encoder.class_names) is not None else None,
        class_values=tuple(values) if (values := encoder.class_values) is not None else None,
    )
    samples = [
        Sample(inputs={"image": torch.rand(3, SIDE, SIDE)}, targets={task_name: encoder.encode(encoder.load(cell))})
        for _ in range(2)
    ]
    return facts, collate_samples(samples)


def test_this_file_reads_the_registry() -> None:
    """A registry that came back empty would make every test below vacuous."""
    assert len(KINDS) >= 11
    assert set(KINDS) - set(TRAINABLE) == {"detection", "multilabel_segmentation"}


@pytest.mark.parametrize("name", STEPPING)
def test_from_two_classes_a_kinds_parts_agree_through_one_step(name: str, tmp_path: Path) -> None:
    """Encoder, head, loss, activation, adapter and default metrics, on the value the encoder produced."""
    kind = task_kind_registry.create(name)
    facts, batch = through_the_kinds_encoder(kind, "label", tmp_path)
    task = a_task(kind=kind, facts=facts)
    backbone = EveryStreamBackbone()
    model = CompositeModel(backbone=backbone, components={task.name: kind.components(task, backbone)})
    model.train()

    result = model.step(batch)

    assert torch.isfinite(result.loss.total), name
    judged = build_metric_sets(
        kind, facts, {label: MetricConfig(**spec) for label, spec in kind.default_metrics.items()}
    )
    if task.name in result.targets:
        judged[Stage.TRAIN].update(result.prediction.outputs[task.name], result.targets[task.name])
        assert set(judged[Stage.TRAIN].compute()) >= set(kind.default_metrics), name


@pytest.mark.parametrize("name", KINDS)
def test_a_kind_draws_or_declines_by_name(name: str) -> None:
    """A kind either serves a reader and a drawer for the samples page, or says why it shows nothing."""
    kind = task_kind_registry.create(name)
    knobs = DrawingKnobs()

    if kind.not_drawn is not None:
        with pytest.raises(TypeError, match=kind.not_drawn):
            kind.annotate(cast("SampleView", None), a_task(kind=kind), torch.zeros(1), torch.zeros(1), 0, knobs)
    else:
        assert kind.reader(knobs) is not None
        assert kind.drawer(knobs) is not None
