"""The standard preprocessor: load, run the stage's pipeline, encode; one object serves every stage and inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from src.core import Batch, DatasetInfo, Geometry, InputInfo, Sample, require_tensor
from src.data import InputEncoder, StandardPreprocessor
from src.data.encoders import MaskEncoder
from src.transforms import SampleTransform
from tests.unit.data.conftest import PreprocessorFactory, pipeline


def test_a_row_becomes_tensors_and_a_missing_target_is_simply_absent(
    preprocessor: StandardPreprocessor, pixels: SampleTransform, row: Sample
) -> None:
    prepared = preprocessor.preprocess(row, pixels)
    unlabeled = preprocessor.preprocess(Sample(inputs={"image": "a.png"}), pixels)

    assert require_tensor(prepared.inputs["image"], name="image").shape == (3, 4, 4)
    assert require_tensor(prepared.targets["species"], name="species").item() == 1
    assert prepared.metadata == {"row": 0} and unlabeled.targets == {}


def test_the_image_and_its_mask_travel_through_one_pipeline_into_their_own_tensors(
    preprocessor: StandardPreprocessor, pixels: SampleTransform, row: Sample
) -> None:
    prepared = preprocessor.preprocess(row, pixels)

    image = require_tensor(prepared.inputs["image"], name="image")
    mask = require_tensor(prepared.targets["mask"], name="mask")
    assert image.shape == (3, 4, 4) and image.dtype is torch.float32
    assert mask.shape == (4, 4) and mask.dtype is torch.long and set(mask.unique().tolist()) <= {0, 1}


def test_a_transform_sees_loaded_values_before_anything_is_encoded(
    make_preprocessor: PreprocessorFactory, mask_encoder: MaskEncoder, row: Sample
) -> None:
    seen: dict[str, object] = {}
    preprocessor = make_preprocessor(auxiliary_inputs={"region": mask_encoder})
    prepare = pipeline(preprocessor)

    def spy(sample: Sample) -> Sample:
        seen["image"] = np.asarray(sample.inputs["image"]).shape
        seen["region"] = np.asarray(sample.auxiliary_inputs["region"]).shape
        seen["species"] = sample.targets["species"]
        return prepare(Sample(inputs=sample.inputs, targets={**sample.targets, "species": "cat"}, metadata={}))

    prepared = preprocessor.preprocess(
        Sample(inputs=row.inputs, targets={"species": "dog"}, auxiliary_inputs={"region": "a_mask.png"}), transform=spy
    )

    assert seen == {"image": (6, 8, 3), "region": (6, 8), "species": "dog"}
    assert require_tensor(prepared.targets["species"], name="species").item() == 0
    assert prepared.auxiliary_inputs == {}


def test_info_gathers_every_encoder_and_geometries_name_what_moves_with_the_image(
    make_preprocessor: PreprocessorFactory, mask_encoder: MaskEncoder
) -> None:
    preprocessor = make_preprocessor(targets={"mask": mask_encoder}, auxiliary_inputs={"region": mask_encoder})

    assert isinstance(preprocessor.info, DatasetInfo) and preprocessor.info.splits == ()
    assert set(preprocessor.info.inputs) == {"image"} and set(preprocessor.info.targets) == {"mask"}
    assert preprocessor.geometries == {
        "inputs": {"image": Geometry.IMAGE},
        "targets": {"mask": Geometry.MASK},
        "auxiliary_inputs": {"region": Geometry.MASK},
    }


def test_collate_delegates_to_the_collator(
    preprocessor: StandardPreprocessor, pixels: SampleTransform, row: Sample
) -> None:
    batch = preprocessor.collate([preprocessor.preprocess(row, pixels), preprocessor.preprocess(row, pixels)])

    assert isinstance(batch, Batch) and len(batch) == 2


def test_an_input_the_row_does_not_carry_is_refused_by_name(preprocessor: StandardPreprocessor) -> None:
    with pytest.raises(KeyError, match="image"):
        preprocessor.preprocess(Sample(inputs={}))


def test_a_image_no_pipeline_prepared_is_refused_before_the_model_sees_it(
    preprocessor: StandardPreprocessor, row: Sample
) -> None:
    with pytest.raises(ValueError, match="transforms"):
        preprocessor.preprocess(row)


def test_an_input_that_is_not_pixels_needs_no_pipeline(make_preprocessor: PreprocessorFactory) -> None:
    prepared = make_preprocessor(inputs={"text": Text()}, targets={}).preprocess(Sample(inputs={"text": "hello"}))

    assert require_tensor(prepared.inputs["text"], name="text").item() == 5


class Text(InputEncoder):
    """A non-pixel input: no pipeline carries it, so its loaded value reaches ``encode`` untouched."""

    @property
    def info(self) -> InputInfo:
        return InputInfo(shape=None, modality="text")

    def encode(self, value: object) -> Tensor:
        return torch.tensor(len(str(value)))
