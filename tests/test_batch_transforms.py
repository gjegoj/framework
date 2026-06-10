"""Unit tests for batch transforms, the scheduling callback, and the soft codec."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.callbacks.batch_transform import BatchTransformCallback
from src.core.entities import Batch
from src.core.ports import BatchTransform
from src.tasks.codecs import MulticlassTaskCodec
from src.tasks.taxonomy import Topology
from src.transforms.batch import CutMix, MixUp, Mosaic, TargetSpec

# ---------------------------------------------------------------- helpers


def _global(key: str, num_classes: int) -> TargetSpec:
    return TargetSpec(key=key, topology=Topology.GLOBAL, num_classes=num_classes)


def _dense(key: str, num_classes: int) -> TargetSpec:
    return TargetSpec(key=key, topology=Topology.DENSE, num_classes=num_classes)


def _trainer(current_epoch: int = 0, max_epochs: int = 10) -> MagicMock:
    t = MagicMock()
    t.current_epoch = current_epoch
    t.max_epochs = max_epochs
    return t


class _AddOneTransform(BatchTransform):
    """Records calls and returns a batch with every input incremented by one."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, batch: Batch) -> Batch:
        self.calls += 1
        return Batch(
            inputs={key: value + 1 for key, value in batch.inputs.items()},
            targets=batch.targets,
            meta=batch.meta,
        )


# ---------------------------------------------------------------- MixUp / CutMix


class TestLabelMixTransforms:
    def test_mixup_keeps_image_shape(self) -> None:
        batch = Batch(inputs={"image": torch.randn(4, 3, 8, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})
        out = MixUp(targets=[_global("label", 3)])(batch)
        assert out.inputs["image"].shape == (4, 3, 8, 8)

    def test_mixup_produces_soft_targets_summing_to_one(self) -> None:
        batch = Batch(inputs={"image": torch.randn(4, 3, 8, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})
        out = MixUp(targets=[_global("label", 3)], alpha=0.5)(batch)
        soft = out.targets["label"]
        assert soft.shape == (4, 3)
        assert soft.dtype == torch.float32
        assert torch.allclose(soft.sum(dim=1), torch.ones(4))

    def test_multihead_two_heads_with_different_num_classes(self) -> None:
        batch = Batch(
            inputs={"image": torch.randn(4, 3, 8, 8)},
            targets={"species": torch.tensor([0, 1, 2, 0]), "anomaly": torch.tensor([1, 0, 1, 1])},
        )
        out = MixUp(targets=[_global("species", 3), _global("anomaly", 2)])(batch)
        assert out.targets["species"].shape == (4, 3)
        assert out.targets["anomaly"].shape == (4, 2)
        # Same sampled mix applied to both heads → both are valid soft distributions.
        assert torch.allclose(out.targets["species"].sum(dim=1), torch.ones(4))
        assert torch.allclose(out.targets["anomaly"].sum(dim=1), torch.ones(4))

    def test_cutmix_produces_soft_targets(self) -> None:
        batch = Batch(inputs={"image": torch.randn(4, 3, 8, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})
        out = CutMix(targets=[_global("label", 3)])(batch)
        assert out.targets["label"].shape == (4, 3)
        assert torch.allclose(out.targets["label"].sum(dim=1), torch.ones(4))

    def test_does_not_mutate_input_batch(self) -> None:
        batch = Batch(inputs={"image": torch.randn(4, 3, 8, 8)}, targets={"label": torch.tensor([0, 1, 2, 0])})
        MixUp(targets=[_global("label", 3)])(batch)
        assert batch.targets["label"].dtype == torch.int64  # original hard labels untouched

    def test_supported_topologies_is_global_only(self) -> None:
        assert MixUp.supported_topologies == frozenset({Topology.GLOBAL})
        assert CutMix.supported_topologies == frozenset({Topology.GLOBAL})


# ---------------------------------------------------------------- Mosaic


class TestMosaic:
    def test_images_only_keeps_shape(self) -> None:
        batch = Batch(inputs={"image": torch.randn(4, 3, 16, 16)}, targets={})
        out = Mosaic(targets=[])(batch)
        assert out.inputs["image"].shape == (4, 3, 16, 16)

    def test_composes_every_dense_mask(self) -> None:
        batch = Batch(
            inputs={"image": torch.randn(4, 3, 16, 16)},
            targets={"mask": torch.randint(0, 5, (4, 16, 16)), "edges": torch.randint(0, 3, (4, 16, 16))},
        )
        out = Mosaic(targets=[_dense("mask", 5), _dense("edges", 3)])(batch)
        assert out.targets["mask"].shape == (4, 16, 16)
        assert out.targets["mask"].dtype == torch.int64
        assert set(out.targets["mask"].unique().tolist()).issubset(set(range(5)))
        assert out.targets["edges"].shape == (4, 16, 16)

    def test_rejects_invalid_center_ratio(self) -> None:
        for bad in [(0.7, 0.3), (0.0, 0.5), (0.3, 1.0)]:
            with pytest.raises(ValueError, match="center_ratio"):
                Mosaic(targets=[], center_ratio=bad)

    def test_supported_topologies_is_dense_only(self) -> None:
        assert Mosaic.supported_topologies == frozenset({Topology.DENSE})


# ---------------------------------------------------------------- BatchTransformCallback


class TestBatchTransformCallback:
    def test_rejects_invalid_fraction(self) -> None:
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError, match="disable_after_fraction"):
                BatchTransformCallback(_AddOneTransform(), disable_after_fraction=bad)

    def test_on_fit_start_resolves_disable_epoch(self) -> None:
        cb = BatchTransformCallback(_AddOneTransform(), disable_after_fraction=0.5)
        cb.on_fit_start(_trainer(max_epochs=20), MagicMock())
        assert cb._disable_epoch == 10

    def test_applies_transform_in_place_while_active(self) -> None:
        fake = _AddOneTransform()
        cb = BatchTransformCallback(fake, disable_after_fraction=1.0)
        cb.on_fit_start(_trainer(max_epochs=10), MagicMock())
        batch = Batch(inputs={"image": torch.zeros(2, 3, 4, 4)}, targets={"label": torch.tensor([0, 1])})

        cb.on_train_batch_start(_trainer(current_epoch=0), MagicMock(), batch, 0)

        assert fake.calls == 1
        assert torch.all(batch.inputs["image"] == 1)  # mutation propagated to the original batch

    def test_skips_transform_after_cutoff(self) -> None:
        fake = _AddOneTransform()
        cb = BatchTransformCallback(fake, disable_after_fraction=0.5)
        cb.on_fit_start(_trainer(max_epochs=10), MagicMock())  # disable_epoch = 5
        batch = Batch(inputs={"image": torch.zeros(2, 3, 4, 4)}, targets={"label": torch.tensor([0, 1])})

        cb.on_train_batch_start(_trainer(current_epoch=7), MagicMock(), batch, 0)

        assert fake.calls == 0
        assert torch.all(batch.inputs["image"] == 0)

    def test_ignores_non_batch_payload(self) -> None:
        fake = _AddOneTransform()
        cb = BatchTransformCallback(fake, disable_after_fraction=1.0)
        cb.on_fit_start(_trainer(max_epochs=10), MagicMock())
        cb.on_train_batch_start(_trainer(current_epoch=0), MagicMock(), {"not": "a batch"}, 0)
        assert fake.calls == 0


# ---------------------------------------------------------------- soft-aware codec


class TestMulticlassCodecSoftTarget:
    def test_soft_target_kept_for_loss_argmax_for_metric(self) -> None:
        soft = torch.tensor([[0.7, 0.3, 0.0], [0.1, 0.2, 0.7]])
        view = MulticlassTaskCodec().adapt(soft)
        assert torch.equal(view.loss, soft)
        assert view.metric.tolist() == [0, 2]

    def test_hard_target_unchanged(self) -> None:
        view = MulticlassTaskCodec().adapt(torch.tensor([0, 2, 1]))
        assert view.loss.dtype == torch.long
        assert torch.equal(view.loss, view.metric)

    def test_hard_column_vector_squeezed(self) -> None:
        view = MulticlassTaskCodec().adapt(torch.tensor([[0], [2], [1]]))
        assert view.loss.tolist() == [0, 2, 1]
