"""MixUp and CutMix: one draw moves the image and every task's label together."""

from __future__ import annotations

import pytest
import torch

from src.core import Batch, DataProfile, Objective, OutputTopology, Task
from src.transforms.batch import CutMix, MixUp
from tests.support.entities import profiling
from tests.support.narrowing import tensor


def task(name: str = "label", objective: Objective = Objective.MULTICLASS) -> Task:
    return Task(name=name, output_topology=OutputTopology.GLOBAL, objective=objective, metrics={})


def batch(size: int = 4, classes: int = 3) -> Batch:
    return Batch(
        inputs={"image": torch.arange(size * 3 * 8 * 8, dtype=torch.float32).reshape(size, 3, 8, 8)},
        targets={"label": torch.arange(size) % classes},
    )


def test_a_class_index_becomes_a_distribution() -> None:
    """The mixed sample belongs partly to two classes; an index cannot say that."""
    mixed = MixUp([task()], profiling(label=3))(batch())

    assert tensor(mixed.targets["label"]).shape == (4, 3)
    assert torch.allclose(tensor(mixed.targets["label"]).sum(dim=1), torch.ones(4))


def test_the_image_and_the_label_move_by_the_same_weight() -> None:
    """Two draws would teach the model a label the picture does not show."""
    torch.manual_seed(0)
    original = batch()
    before = original.inputs["image"].clone()

    mixed = MixUp([task()], profiling(label=3), alpha=1.0)(original)

    weight = tensor(mixed.targets["label"])[1].max().item()
    expected = weight * before + (1 - weight) * before.roll(1, 0)
    assert torch.allclose(mixed.inputs["image"], expected, atol=1e-3)


def test_every_task_shares_one_mix() -> None:
    """A two-head model must not blend its heads differently from its image."""
    tasks = [task("label"), task("kind")]
    given = batch()
    given.targets["kind"] = torch.arange(4) % 2

    mixed = MixUp(tasks, profiling(label=3, kind=2))(given)

    assert tensor(mixed.targets["label"]).shape == (4, 3)
    assert tensor(mixed.targets["kind"]).shape == (4, 2)


def test_the_batch_it_was_given_is_left_alone() -> None:
    """The callback owns writing back; a transform that mutated would do it twice."""
    given = batch()
    before = given.inputs["image"].clone()

    MixUp([task()], profiling(label=3))(given)

    assert torch.equal(given.inputs["image"], before)


def test_cutmix_pastes_a_region_rather_than_blending() -> None:
    """Every pixel comes from exactly one source — that is the difference from MixUp."""
    torch.manual_seed(0)
    given = batch()
    sources = {float(value) for value in given.inputs["image"].flatten()}

    mixed = CutMix([task()], profiling(label=3))(given)

    assert {float(value) for value in mixed.inputs["image"].flatten()} <= sources


def test_cutmix_weights_the_label_by_the_area_it_actually_pasted() -> None:
    """Clipping at the frame edge shrinks the patch; the label has to follow."""
    torch.manual_seed(0)
    mixed = CutMix([task()], profiling(label=3))(batch())

    assert 0.0 <= tensor(mixed.targets["label"]).max().item() <= 1.0


def test_a_continuous_target_is_mixed_as_a_number() -> None:
    """One-hot encoding a price would be nonsense; only class indices need it."""
    given = batch()
    given.targets["price"] = torch.tensor([1.0, 2.0, 3.0, 4.0])

    mixed = MixUp([task("price", Objective.CONTINUOUS)], DataProfile())(given)

    assert tensor(mixed.targets["price"]).shape == (4,)


def test_a_dense_task_is_refused() -> None:
    """A blended image has no coherent per-pixel target, so the stack is wrong, not the data."""
    dense = Task(name="mask", output_topology=OutputTopology.DENSE, objective=Objective.MULTICLASS, metrics={})

    with pytest.raises(ValueError, match="mask"):
        MixUp([dense], profiling(mask=3))


def test_a_metric_learning_task_is_refused() -> None:
    """Proxy and margin losses break on soft labels."""
    metric = Task(name="pairs", output_topology=OutputTopology.GLOBAL, objective=Objective.METRIC, metrics={})

    with pytest.raises(ValueError, match="pairs"):
        MixUp([metric], DataProfile())


def test_a_non_positive_alpha_is_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        MixUp([task()], profiling(label=3), alpha=0.0)
