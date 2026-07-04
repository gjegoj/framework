"""Custom Albumentations augmentations: label-aware rotation and border crop."""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest

from src.core.entities import Sample
from src.data import (
    AlbumentationsTransform,
    LabelEncoder,
)


class TestRotate90WithLabel:
    def test_image_and_label_share_the_same_k(self) -> None:
        """The sampled quarter-turn drives both the image and the label, so they always agree."""
        from src.transforms import Rotate90WithLabel

        transform = Rotate90WithLabel(label_key="rotation", p=1.0)
        image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)  # asymmetric → rotation observable
        for k in range(4):
            assert np.array_equal(transform.apply(image, k=k), np.rot90(image, k))
            assert transform.apply_to_label(0, k=k) == k

    def test_label_wraps_modulo_four(self) -> None:
        from src.transforms import Rotate90WithLabel

        transform = Rotate90WithLabel(label_key="rotation", p=1.0)
        assert transform.apply_to_label(2, k=3) == 1  # (2 + 3) % 4
        assert transform.apply_to_label(3, k=1) == 0  # wraps

    def test_compose_threads_rotation_label_as_int(self) -> None:
        from src.transforms import Rotate90WithLabel

        compose = A.Compose([Rotate90WithLabel(label_key="rotation", p=1.0)], seed=0)
        result = compose(image=np.zeros((4, 4, 3), np.uint8), rotation=0)
        assert isinstance(result["rotation"], int)
        assert result["rotation"] in {0, 1, 2, 3}

    def test_label_key_is_configurable(self) -> None:
        """The bound data key is a parameter, so the transform is not tied to a task named 'rotation'."""
        from src.transforms import Rotate90WithLabel

        compose = A.Compose([Rotate90WithLabel(label_key="orientation", p=1.0)], seed=0)
        result = compose(image=np.zeros((4, 4, 3), np.uint8), orientation=0)
        assert "orientation" in result
        assert result["orientation"] in {0, 1, 2, 3}

    def test_end_to_end_label_encoder_index_is_bumped_then_tensorized(self) -> None:
        """Encoder.load yields the class index, the aug bumps it, to_tensor wraps the result."""
        from src.transforms import Rotate90WithLabel

        encoder = LabelEncoder(class_mapping={0: "0", 1: "90", 2: "180", 3: "270"})
        transform = AlbumentationsTransform(A.Compose([Rotate90WithLabel(label_key="rotation", p=1.0)], seed=1))
        sample = Sample(inputs={"image": np.zeros((4, 4, 3), np.uint8)}, targets={"rotation": encoder.load("0")})
        index = transform.apply(sample).targets["rotation"]
        assert index in {0, 1, 2, 3}
        assert encoder.to_tensor(index).item() == index


class TestRandomBorderCropWithLabel:
    def test_crop_shrinks_image_and_mask_together(self) -> None:
        """The same sampled crop applies to image and mask, so they stay aligned."""
        from src.transforms import RandomBorderCropWithLabel

        compose = A.Compose([RandomBorderCropWithLabel(min_crop_threshold=0.1, p=1.0)], seed=0)
        result = compose(image=np.zeros((100, 120, 3), np.uint8), mask=np.zeros((100, 120), np.uint8), label=0)
        cropped_height, cropped_width = result["image"].shape[:2]
        assert (cropped_height, cropped_width) < (100, 120)
        assert result["mask"].shape[:2] == (cropped_height, cropped_width)

    def test_applying_the_crop_sets_the_label_to_one(self) -> None:
        from src.transforms import RandomBorderCropWithLabel

        transform = RandomBorderCropWithLabel(p=1.0)
        assert transform.apply_to_label(0) == 1
        assert transform.apply_to_label(7) == 1  # the prior value is irrelevant

    def test_threshold_is_guaranteed_on_at_least_one_side(self) -> None:
        """min_crop_threshold forces the strongest crop to reach the bound on every draw."""
        from src.transforms import RandomBorderCropWithLabel

        threshold = 0.15
        transform = RandomBorderCropWithLabel(
            crop_left=0.2, crop_right=0.2, crop_top=0.2, crop_bottom=0.2, min_crop_threshold=threshold
        )
        height, width = 100, 120
        for _ in range(200):
            x_min, y_min, x_max, y_max = transform.get_params_dependent_on_data({"shape": (height, width, 3)}, {})[
                "crop_coords"
            ]
            strongest = max(x_min / width, 1 - x_max / width, y_min / height, 1 - y_max / height)
            assert strongest >= threshold

    def test_threshold_above_largest_limit_is_rejected(self) -> None:
        from src.transforms import RandomBorderCropWithLabel

        with pytest.raises(ValueError, match="min_crop_threshold"):
            RandomBorderCropWithLabel(
                crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, min_crop_threshold=0.2
            )

    def test_label_key_is_configurable(self) -> None:
        from src.transforms import RandomBorderCropWithLabel

        compose = A.Compose([RandomBorderCropWithLabel(p=1.0, label_key="defect")], seed=0)
        result = compose(image=np.zeros((40, 40, 3), np.uint8), defect=0)
        assert result["defect"] == 1
