"""The cache reaches loaders and encoders from the composition root, or not at all."""

from __future__ import annotations

from src.assembly.data import build_cache, build_data_schema
from src.data import RamCache
from tests.support.configs import DATA, paper_config

RAM = {"cache": {"name": "ram", "max_gib": 0.5}}


def test_no_cache_section_means_no_cache() -> None:
    assert build_cache(paper_config()) is None


def test_the_declared_cache_is_built_from_the_registry() -> None:
    cache = build_cache(paper_config(data=DATA | RAM))

    assert isinstance(cache, RamCache)


def test_input_loaders_are_wrapped_when_a_cache_is_given() -> None:
    """Wrapping is visible at the composition root rather than hidden in a loader."""
    plain = build_data_schema(paper_config()).inputs["image"].loader
    wrapped = build_data_schema(paper_config(), RamCache(max_gib=0.5)).inputs["image"].loader

    assert wrapped is not plain


def test_a_mask_encoder_is_offered_the_cache() -> None:
    """It reads files behind a loader of its own, so it takes the cache instead of being wrapped."""
    config = paper_config(
        tasks={
            "mask": {
                "preset": "segmentation",
                "target": "mask",
                "target_encoder": {"name": "mask", "num_classes": 2},
            }
        },
        model={"name": "smp", "architecture": "unet", "encoder_name": "resnet18"},
    )

    encoder = build_data_schema(config, RamCache(max_gib=0.5)).targets["mask"].encoder

    assert encoder.num_classes == 2


def test_an_encoder_that_does_not_want_a_cache_is_not_handed_one() -> None:
    """The derived value is offered, not forced; a label encoder names no 'cache'."""
    schema = build_data_schema(paper_config(), RamCache(max_gib=0.5))

    assert schema.targets["label"].encoder.class_names is None


def test_a_run_without_a_cache_builds_exactly_as_before() -> None:
    schema = build_data_schema(paper_config())

    assert schema.inputs["image"].column == "image"
    assert schema.targets["label"].encoder.num_classes is None
