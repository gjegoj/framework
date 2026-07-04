"""Per-sample input transforms: identity (embedding modality) and multi-view."""

from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from src.core.entities import Sample


class TestIdentityTransform:
    def test_tensorizes_input_vector(self) -> None:
        from src.transforms.sample import IdentityTransform

        sample = Sample(inputs={"embedding": np.arange(4, dtype=np.float32)})
        result = IdentityTransform().apply(sample)
        assert isinstance(result.inputs["embedding"], torch.Tensor)
        assert result.inputs["embedding"].dtype == torch.float32
        assert result.inputs["embedding"].shape == (4,)

    def test_passes_targets_through_unchanged(self) -> None:
        from src.transforms.sample import IdentityTransform

        target = torch.tensor(2)
        sample = Sample(inputs={"embedding": np.zeros(4, dtype=np.float32)}, targets={"label": target})
        result = IdentityTransform().apply(sample)
        assert result.targets["label"] is target


class TestMultiViewTransform:
    @staticmethod
    def _aug() -> A.Compose:
        # a randomized geometric op makes independent-vs-shared sampling observable
        return A.Compose([A.Affine(rotate=(-45, 45), p=1.0), A.Resize(8, 8), A.Normalize(), ToTensorV2()], seed=0)

    @staticmethod
    def _two_views() -> Sample:
        image = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        return Sample(inputs={"a": image.copy(), "b": image.copy()})  # identical content in both views

    def test_shared_applies_identical_params_to_all_views(self) -> None:
        from src.transforms.sample import MultiViewTransform

        result = MultiViewTransform(self._aug(), shared=True).apply(self._two_views())
        assert torch.equal(result.inputs["a"], result.inputs["b"])  # one sampling → identical tensors

    def test_independent_samples_each_view_separately(self) -> None:
        from src.transforms.sample import MultiViewTransform

        result = MultiViewTransform(self._aug(), shared=False).apply(self._two_views())
        assert not torch.equal(result.inputs["a"], result.inputs["b"])  # per-view sampling → differ

    def test_every_view_is_tensorized(self) -> None:
        from src.transforms.sample import MultiViewTransform

        for shared in (True, False):
            result = MultiViewTransform(self._aug(), shared=shared).apply(self._two_views())
            assert result.inputs["a"].shape == (3, 8, 8)
            assert result.inputs["b"].shape == (3, 8, 8)

    def test_shared_single_input_does_not_break(self) -> None:
        from src.transforms.sample import MultiViewTransform

        sample = Sample(inputs={"image": np.zeros((16, 16, 3), dtype=np.uint8)})
        result = MultiViewTransform(self._aug(), shared=True).apply(sample)
        assert result.inputs["image"].shape == (3, 8, 8)
