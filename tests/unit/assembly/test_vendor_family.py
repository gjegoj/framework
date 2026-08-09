"""One name decides the family, and what it cannot serve is refused before the first epoch."""

from __future__ import annotations

from typing import Any

import pytest

from src.assembly.vendor import is_vendor_family, refuse_what_a_vendor_cannot_serve
from src.config import ExperimentConfig
from src.data.registry import vendor_data_module_registry
from src.models.registry import model_registry
from tests.support.configs import MODEL, TASKS, paper_config

VENDOR = {"name": "yolo", "model_name": "yolov8n.yaml"}
DETECTION = {"boxes": {"preset": "detection"}}
DESCRIPTOR = {"source": "coco8.yaml", "inputs": {}}
"""What a vendor pipeline's data section is: a descriptor, and no columns of our own."""


def experiment(model: dict[str, Any] = VENDOR, tasks: dict[str, Any] = DETECTION, **overrides: Any) -> ExperimentConfig:
    """A detection run by default, and whatever section a test adds to it."""
    return paper_config(data=DESCRIPTOR, tasks=tasks, model=model, **overrides)


def test_the_model_section_is_what_says_which_family_a_run_is() -> None:
    """One reading of one key decides the model *and* the data pipeline.

    Read in two places from two keys, the halves could disagree about what kind of run
    this is — a table pipeline feeding a vendor model, and nothing saying so.
    """
    assert is_vendor_family(experiment())
    assert not is_vendor_family(experiment(model=MODEL, tasks=TASKS))


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({"transforms": {"train": {"_target_": "src.transforms.AlbumentationsTransform"}}}, "box-aware"),
        ({"export": [{"name": "torchscript"}]}, "'export'"),
        ({"adapters": {"name": "lora"}}, "'adapters'"),
    ],
)
def test_a_section_a_vendor_family_cannot_serve_is_refused_by_name(section: dict[str, Any], expected: str) -> None:
    """Failing at assembly beats failing an hour into training, and a section silently
    ignored is worse than either: the run reports numbers for a recipe nobody ran, and
    the difference only surfaces when somebody tries to reproduce it.
    """
    with pytest.raises(ValueError, match=expected):
        refuse_what_a_vendor_cannot_serve(experiment(**section))


def test_the_callback_that_blends_targets_is_refused() -> None:
    """A batch transform rewrites each task's label, and a vendor family's are objects."""
    declared = experiment(callbacks=[{"name": "batch_transform", "transform": {"_target_": "src.transforms.MixUp"}}])

    with pytest.raises(ValueError, match="batch_transform"):
        refuse_what_a_vendor_cannot_serve(declared)


def test_a_vendor_family_serves_exactly_one_task() -> None:
    """Its head is built for one. A second would train nothing and report nothing, and the
    tracker would show a task with no numbers under it and no reason given.
    """
    two = experiment(tasks={"boxes": {"preset": "detection"}, "extra": {"preset": "detection"}})

    with pytest.raises(ValueError, match="one task"):
        refuse_what_a_vendor_cannot_serve(two)


def test_our_bricks_declared_on_its_task_are_refused() -> None:
    """A vendor's assigner, loss and decoding are one design; half-replacing it silently
    produces a model that trains against a different objective than it reports.
    """
    declared = experiment(tasks={"boxes": {"preset": "detection", "loss": {"name": "cross_entropy"}}})

    with pytest.raises(ValueError, match="builds its own"):
        refuse_what_a_vendor_cannot_serve(declared)


def test_vendor_families_bring_both_halves() -> None:
    """A family that arrives whole brings its network *and* the pipeline that feeds it.

    The two live in packages that do not import one another, so the key is spelled twice
    — once in ``model_registry``, once in ``vendor_data_module_registry``. Registered as a
    model only, a run would compose its network and then fail looking for a pipeline; this
    is what turns "two things to keep in step" into one that a test keeps.
    """
    pytest.importorskip("ultralytics", reason="the only vendor family shipped is optional")
    import src.data
    import src.models  # noqa: F401

    assert set(vendor_data_module_registry) <= set(model_registry)


def test_a_refusal_names_the_family_that_cannot_serve_the_section() -> None:
    """ "A vendor family cannot..." states as universal what is one family's limit.

    Two are shipped-family facts rather than vendor facts — a box-aware pipeline, a head
    built for one task — and a second family will disagree with at least one of them. The
    sentence names which model is refusing, so the reader knows whose rule they hit.
    """
    with pytest.raises(ValueError, match="'yolo'"):
        refuse_what_a_vendor_cannot_serve(experiment(export=[{"name": "torchscript"}]))


def test_a_composed_run_is_left_alone() -> None:
    """None of this applies to the family that composes, so none of it is checked for it."""
    composed = paper_config(model=MODEL, tasks=TASKS, export=[{"name": "torchscript"}])

    refuse_what_a_vendor_cannot_serve(composed)
