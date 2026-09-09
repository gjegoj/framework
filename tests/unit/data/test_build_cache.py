"""The cache reaches loaders and encoders from the composition root, or not at all."""

from __future__ import annotations

from pathlib import Path

from src.data import LoaderCache, RamCache
from src.data.build import build_cache
from tests.support.configs import DATA, paper_config, schema_of

RAM = {"cache": {"name": "ram", "max_gib": 0.5}}


def test_no_cache_section_means_no_cache() -> None:
    assert build_cache(paper_config().data.cache) is None


def test_the_declared_cache_is_built_from_the_registry() -> None:
    cache = build_cache(paper_config(data=DATA | RAM).data.cache)

    assert isinstance(cache, RamCache)


def test_input_loaders_are_wrapped_when_a_cache_is_given() -> None:
    """Wrapping is visible at the composition root rather than hidden in a loader."""
    plain = schema_of(paper_config()).inputs["image"].loader
    wrapped = schema_of(paper_config(), RamCache(max_gib=0.5)).inputs["image"].loader

    assert wrapped is not plain


def test_a_mask_encoder_reads_through_the_cache(tmp_path: Path) -> None:
    """It reads files behind a loader of its own, so it is told to read through the cache rather than wrapped."""
    import numpy as np
    from PIL import Image

    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / "m.png")
    config = paper_config(
        tasks={
            "mask": {
                "kind": "segmentation",
                "target": "mask",
                "classes": {0: "a", 1: "b"},
                "target_encoder": {"name": "mask", "root": str(tmp_path)},
            }
        },
        model={"name": "smp", "architecture": "unet", "encoder_name": "resnet18"},
    )
    cache = RamCache(max_gib=0.5)

    encoder = schema_of(config, cache).targets["mask"].encoder
    cache.warm(["m.png"], encoder.load)

    assert cache.usage().files == 1  # the warm-up went through the encoder's own read, into the shared store


def test_an_encoder_that_reads_no_files_has_nothing_to_cache() -> None:
    """A label encoder is not a ``FileTargetEncoder``, so the cache never reaches it."""
    schema = schema_of(paper_config(), RamCache(max_gib=0.5))

    assert schema.targets["label"].encoder.class_names == ["cat", "dog"]


def test_a_run_without_a_cache_builds_exactly_as_before() -> None:
    schema = schema_of(paper_config())

    assert schema.inputs["image"].column == "image"
    assert schema.targets["label"].encoder.num_classes == 2


def test_pipeline_namespaces_are_the_schemas_own_columns() -> None:
    """The pin between two hand-written copies of one name.

    The pipeline builder scopes each column's cache namespace while the columns are being
    constructed — before a schema exists — so it cannot call
    ``DataSchema.columns_by_role`` and names the pairs itself. This is what
    keeps the two walks from silently filing a bar's title and a store's keys
    under different columns.
    """

    class RecordingCache(RamCache):
        def __init__(self) -> None:
            super().__init__(max_gib=0.5)
            self.namespaces: list[tuple[str, ...]] = []

        def scoped(self, *namespace: str) -> LoaderCache:
            self.namespaces.append(namespace)
            return super().scoped(*namespace)

    spy = RecordingCache()
    config = paper_config(
        data=DATA | {"auxiliary_inputs": {"lesion": {"column": "mask"}}, "cache": {"name": "ram", "max_gib": 0.5}}
    )

    schema = schema_of(config, spy)

    assert sorted(spy.namespaces) == sorted((str(role), name) for role, name, _ in schema.columns_by_role())
