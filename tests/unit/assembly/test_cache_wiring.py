"""The cache reaches loaders and encoders from the composition root, or not at all."""

from __future__ import annotations

from src.assembly.data import build_cache, build_data_schema
from src.data import LoaderCache, RamCache
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


def test_assembly_namespaces_and_schema_labels_are_one_spelling() -> None:
    """The pin between two hand-written copies of one name.

    Assembly scopes each column's cache namespace while the columns are being
    constructed — before a schema exists — so it cannot call
    ``DataSchema.labelled_columns`` and spells the labels itself. This is what
    keeps a typo in either spelling from silently filing a bar's label and a
    store's keys under different names.
    """

    class RecordingCache(RamCache):
        def __init__(self) -> None:
            super().__init__(max_gib=0.5)
            self.namespaces: list[str] = []

        def scoped(self, namespace: str) -> LoaderCache:
            self.namespaces.append(namespace)
            return super().scoped(namespace)

    spy = RecordingCache()
    config = paper_config(
        data=DATA | {"auxiliary_inputs": {"lesion": {"column": "mask"}}, "cache": {"name": "ram", "max_gib": 0.5}}
    )

    schema = build_data_schema(config, spy)

    assert sorted(spy.namespaces) == sorted(label for label, _ in schema.labelled_columns())
