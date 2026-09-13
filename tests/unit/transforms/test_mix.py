"""Two samples become one image, and every task's label is rewritten by the same draw."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch
from torch import Tensor

from src.core import Batch, TargetInfo
from src.tasks import MetricLearning, Task
from src.tasks.registry import task_registry
from src.transforms import BatchTransform, CutMix, MixUp

CLASSES = {0: "cat", 1: "dog"}
PICTURES = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
"""Two images a mix can be read off exactly: one all zeros, one all ones."""

DRAWS = 20
"""Enough draws that a run of them cannot all round their patch away; see the test that says why."""


def task(kind: str = "classification", name: str = "label", classes: Mapping[int, str] | None = CLASSES) -> Task:
    return task_registry.get(kind)(name, TargetInfo(classes=classes))


def batch(**targets: Tensor) -> Batch:
    return Batch(inputs={"image": PICTURES.clone()}, targets=dict(targets), count=2)


def mixed(transform: BatchTransform, tasks: Sequence[Task], **targets: Tensor) -> Batch:
    return transform.for_tasks(tasks)(batch(**targets))


def weights_of(image: Tensor) -> tuple[float, float]:
    """The two shares the mix was made with, read off a blend of a constant zero and a constant one.

    Both read off the result rather than one of them subtracted from the other: a weight near 1 leaves
    a complement of a few ten-thousandths, and subtracting it back costs most of its significant digits.
    """
    return float(image[1].mean()), float(image[0].mean())


class TestMixUp:
    def test_the_image_is_blended_with_its_neighbour_by_one_weight(self) -> None:
        """Every pixel of a blend of a constant zero and a constant one sums to exactly one."""
        result = mixed(MixUp(), [task()], label=torch.tensor([0, 1]))

        image = result.inputs["image"]
        assert isinstance(image, Tensor)
        assert torch.allclose(image[0] + image[1], torch.ones(3, 8, 8))

    def test_the_label_is_mixed_by_the_weight_the_image_was(self) -> None:
        """One draw, or an image would take one neighbour's pixels and another's label."""
        result = mixed(MixUp(), [task()], label=torch.tensor([0, 1]))

        kept, taken = weights_of(torch.as_tensor(result.inputs["image"]))
        assert torch.allclose(torch.as_tensor(result.targets["label"]), torch.tensor([[kept, taken], [taken, kept]]))

    def test_every_task_is_rewritten_from_the_same_draw(self) -> None:
        tasks = [task(name="label"), task(kind="regression", name="age")]

        result = mixed(MixUp(), tasks, label=torch.tensor([0, 1]), age=torch.tensor([0.0, 10.0]))

        kept, taken = weights_of(torch.as_tensor(result.inputs["image"]))
        assert torch.allclose(torch.as_tensor(result.targets["age"]), torch.tensor([10.0 * taken, 10.0 * kept]))

    def test_a_class_index_becomes_the_share_of_each_class_it_stands_for(self) -> None:
        """An index is not a thing to average: the kinds whose targets are one widen them first."""
        result = mixed(MixUp(), [task()], label=torch.tensor([0, 1]))

        assert torch.as_tensor(result.targets["label"]).shape == (2, 2)

    def test_the_batch_it_was_handed_is_never_written_into(self) -> None:
        given = batch(label=torch.tensor([0, 1]))

        MixUp().for_tasks([task()])(given)

        assert torch.equal(torch.as_tensor(given.inputs["image"]), PICTURES)


class TestCutMix:
    def test_a_patch_of_the_neighbour_is_pasted_in(self) -> None:
        """Whole pixels, never a blend — and across a run of draws, a patch that is really there.

        Across draws rather than on one, because the patch is sized from the draw: a weight above
        0.94 rounds an 8-pixel side away to nothing, which is a legitimate outcome about once in
        sixteen. Asserting it on a single draw is asserting on a particular random value, and what
        makes that pass today is only how many draws the tests before it happened to take.
        """
        pasted = 0
        for _ in range(DRAWS):
            image = torch.as_tensor(mixed(CutMix(), [task()], label=torch.tensor([0, 1])).inputs["image"])
            assert set(image[0].unique().tolist()) <= {0.0, 1.0}, "pasted whole, never blended"
            pasted += int(image[0].any())

        assert pasted, "no draw pasted anything at all"

    def test_the_label_weight_is_the_area_that_stayed(self) -> None:
        """The pasted area is what the weight has to mean, or the label describes an image nobody saw."""
        result = mixed(CutMix(), [task()], label=torch.tensor([0, 1]))

        pasted = float(torch.as_tensor(result.inputs["image"])[0].mean())
        assert float(torch.as_tensor(result.targets["label"])[0][0]) == pytest.approx(1.0 - pasted)


class TestBinding:
    def test_a_task_measured_at_every_pixel_is_refused_by_name(self) -> None:
        """A blended image has no coherent per-pixel target, and this is known before the first batch."""
        with pytest.raises(ValueError, match="mask"):
            MixUp().for_tasks([task(), task(kind="segmentation", name="mask")])

    def test_a_task_answering_with_a_direction_is_refused_by_name(self) -> None:
        """An average of two identities names a third that neither sample was, and nothing would say so."""
        identity = MetricLearning("identity", TargetInfo(classes=CLASSES), embedding_dim=4)

        with pytest.raises(ValueError, match="identity"):
            MixUp().for_tasks([task(), identity])

    def test_binding_leaves_the_declared_transform_as_it_was(self) -> None:
        """The binding lives in what comes back, not in the object: what a declaration built stays what
        it declared, so one of them serves two runs and neither can reach into the other's tasks."""
        declared = MixUp()

        first = declared.for_tasks([task()])
        second = declared.for_tasks([task(name="other")])

        assert set(first(batch(label=torch.tensor([0, 1]))).targets) == {"label"}
        assert set(second(batch(other=torch.tensor([0, 1]))).targets) == {"other"}

    @pytest.mark.parametrize("transform", [MixUp, CutMix])
    def test_it_is_the_seam_a_callback_applies(self, transform: type) -> None:
        assert isinstance(transform(), BatchTransform)

    @pytest.mark.parametrize("alpha", [0.0, -1.0], ids=repr)
    def test_a_draw_with_no_spread_is_refused_where_it_was_declared(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            MixUp(alpha=alpha)
