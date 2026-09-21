"""Every shipped YAML composes and validates: the examples cannot drift from the schema."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, cast

import numpy as np
import pytest

from src.config import ComponentConfig, ExperimentConfig, WeightedLossConfig, load_config
from src.config.instantiate import resolve_factory
from src.core import Geometry, Registry, Sample, require_tensor
from src.data.registry import (
    data_module_registry,
    input_encoder_registry,
    preprocessor_registry,
    target_encoder_registry,
)
from src.export.backends.onnx import OnnxExporter
from src.export.backends.tensorrt import TensorRtExporter
from src.export.build import build_exporters
from src.export.registry import exporter_registry
from src.losses.registry import loss_registry
from src.metrics.registry import metric_registry
from src.models.build import NATIVE
from src.models.registry import adapter_registry, backbone_registry, head_registry, model_registry, neck_registry
from src.tasks.registry import task_registry
from src.tracking.registry import tracker_registry
from src.training.registry import learner_registry, optimizer_registry, scheduler_registry
from src.transforms.build import build_transforms
from tests.support.declarations import EXAMPLES, composed


def test_the_examples_a_reader_is_sent_to_are_in_the_repository() -> None:
    """Named rather than counted: these are the lines README and `make test-run` tell a reader to run.

    A glob is the subject of the table below, so an example missing from a fresh clone — left
    untracked, or ignored by a rule meant for somewhere else — would empty that table rather than
    fail it, and every check on the shipped examples would pass on nothing.
    """
    assert {"classification", "segmentation", "finetuning", "multitask"} <= set(EXAMPLES)


@pytest.mark.parametrize("example", EXAMPLES)
def test_a_shipped_example_composes_into_a_valid_experiment(example: str) -> None:
    config = load_config(composed(f"experiment=examples/{example}"))

    assert isinstance(config, ExperimentConfig) and config.tasks


@pytest.mark.parametrize("group", ["default", "augmented"])
def test_a_shipped_transforms_group_prepares_a_image_for_the_declared_size(group: str) -> None:
    """The pixel pipeline lives in YAML, so a wrong import path or a dropped tensor step must fail here."""
    config = load_config(composed("experiment=examples/classification", f"transforms={group}"))
    geometries = {"inputs": {"image": Geometry.IMAGE}, "targets": {"mask": Geometry.MASK}, "auxiliary_inputs": {}}

    built = build_transforms(config.transforms, geometries)

    sample = Sample(inputs={"image": np.zeros((6, 8, 3), np.uint8)}, targets={"mask": np.zeros((6, 8), np.int64)})
    for stage, prepare in built.items():
        prepared = prepare(sample)
        image = require_tensor(prepared.inputs["image"], name=stage)
        assert image.shape == (3, 224, 224) and image.dtype.is_floating_point
        assert require_tensor(prepared.targets["mask"], name="mask").shape == (224, 224)


def drawn_at(seed: int) -> list[float]:
    """What the shipped augmented chain brightens one grey image to, four images in a row."""
    config = load_config(composed("experiment=examples/classification", "transforms=augmented", f"seed={seed}"))
    geometries = {"inputs": {"image": Geometry.IMAGE}, "targets": {}, "auxiliary_inputs": {}}
    prepare = build_transforms(config.transforms, geometries)["train"]
    grey = Sample(inputs={"image": np.full((32, 32, 3), 128, np.uint8)}, targets={})
    return [float(require_tensor(prepare(grey).inputs["image"], name="train")[0, 0, 0]) for _ in range(4)]


def test_the_shipped_augmentations_draw_what_the_runs_seed_settles() -> None:
    """A run repeated at one seed trains on the same images, and another seed gives another run.

    The chain has to say `seed: ${seed}` for this: measured on albumentationsx 2.3.7, a pipeline draws
    from a generator of its own that `seed_everything` never reaches, so an interpolation dropped from
    the group composes perfectly and leaves a run on the shipped loader unrepeatable.
    """
    assert drawn_at(123) == drawn_at(123)
    assert drawn_at(123) != drawn_at(321)


@pytest.mark.parametrize("group", ["onnx", "pt2", "torchscript", "tensorrt", "ncnn", "all"])
def test_a_shipped_export_group_builds_what_it_declares(group: str) -> None:
    """A name that resolves is not a declaration that works: an option spelled wrong reaches `params`
    untouched and answers for it at the end of a run, with weights already trained and nothing shipped.

    Building is the only thing that reads those options, and the two formats no machine here can run are
    built too — everything they can be wrong about is settled before their libraries are looked for.
    """
    config = load_config(composed("experiment=examples/classification", f"export={group}"))

    built = build_exporters(config.export)

    assert [one.suffix for one in built] == [suffix for suffix, _ in SHIPPED[group]]
    for one, (_, declared) in zip(built, SHIPPED[group], strict=True):
        assert {option: getattr(one, option) for option in declared} == declared


OPSET = 18
"""The operator set every shipped ONNX declaration pins, wherever it is written."""


SHIPPED: Mapping[str, list[tuple[str, Mapping[str, object]]]] = {
    "onnx": [("onnx", {"opset": OPSET, "simplify": True})],
    "pt2": [("pt2", {})],
    "torchscript": [("pt", {})],
    "tensorrt": [("engine", {"precision": "fp32", "min_batch": 1, "opt_batch": 1, "max_batch": 1})],
    "ncnn": [("param", {"fp16": False})],
    "all": [("onnx", {"opset": OPSET, "simplify": True}), ("pt2", {}), ("pt", {})],
}
"""What each shipped group writes and under what, named here so a group quietly losing either is a failure.

The suffix alone would not notice an option: a group is read by whoever declares `export=`, and every
option in it changes the artifact rather than the name it lands under.
"""


@pytest.mark.parametrize("group", ["onnx", "all", "tensorrt"])
def test_every_shipped_group_that_writes_onnx_writes_the_same_onnx(group: str) -> None:
    """One format declared in three files: on its own, among `all`, and under the engine compiled from it.

    Written three times because a group is a whole declaration and none of them can reach into another —
    so nothing but this holds them together, and `export=all` shipping a different graph than `export=onnx`
    is a difference nobody would see until two deployments disagreed.
    """
    config = load_config(composed("experiment=examples/classification", f"export={group}"))

    built = build_exporters(config.export)
    written = [one for one in built if isinstance(one, OnnxExporter)]
    written += [one.onnx for one in built if isinstance(one, TensorRtExporter)]

    assert [(one.opset, one.simplify) for one in written] == [(OPSET, True)]


def declared_names(config: ExperimentConfig) -> Iterator[tuple[ComponentConfig, Registry[Any] | None]]:
    """Every name a shipped file writes, beside the registry that has to hold it.

    A transform names no registry: a pixel pipeline is a list of foreign objects reached by import path,
    which `resolve_factory` resolves just the same.
    """
    yield config.model, model_registry
    if config.model.backbone is not None:
        yield config.model.backbone, backbone_registry
    if config.model.neck is not None:
        yield config.model.neck, neck_registry

    yield config.preprocessing, preprocessor_registry
    yield from ((one, input_encoder_registry) for one in (config.preprocessing.inputs or {}).values())
    yield from ((one, None) for one in config.transforms.values())
    yield from ((one, exporter_registry) for one in config.export)
    yield config.data, data_module_registry
    yield config.learner, learner_registry
    yield config.optimizer, optimizer_registry
    if config.scheduler is not None:
        yield config.scheduler, scheduler_registry
    if config.tracker is not None:
        yield config.tracker, tracker_registry
    if config.adapter is not None:
        yield config.adapter, adapter_registry
    for task in config.tasks.values():
        yield task.kind, task_registry
        if task.target_encoder is not None:
            yield task.target_encoder, target_encoder_registry
        if task.head is not None and task.head.name != NATIVE:
            yield task.head, head_registry
        for one in task.loss if isinstance(task.loss, list) else [task.loss]:
            declared = one.loss if isinstance(one, WeightedLossConfig) else one
            if declared is not None:
                yield declared, loss_registry
        yield from ((one, metric_registry) for one in (task.metrics or {}).values())


def unresolved_names(config: ExperimentConfig) -> list[str]:
    """Every name this configuration writes that nothing implements, with the reason each failed."""
    unresolved = []
    for declared, registry in declared_names(config):
        try:
            resolve_factory(declared, registry)
        except (LookupError, TypeError) as error:
            unresolved.append(f"{declared.spelled}: {error}")
    return unresolved


def test_a_neck_a_file_declares_is_held_to_the_registry_that_serves_it() -> None:
    """A position left out of the walk above is a name nothing checks, and it fails in silence: the walk
    yields no pair for it, so there is nothing to resolve and nothing to report. Written as a name
    nothing implements, because a walk that skipped the position would pass on such a name just as
    happily as on one the registry holds.

    ``nonesuch`` rather than a misspelling of ``projector``: the ``typos`` hook rewrites a word that has
    exactly one candidate correction, and writes without saying so. Measured here — it turned this very
    declaration into the spelling that resolves, and the gate that had just been green went red.
    """
    config = load_config(composed("experiment=examples/classification", "+model.neck={name: nonesuch, width: 8}"))

    assert [name for name in unresolved_names(config) if name.startswith("nonesuch:")]


@pytest.mark.parametrize(
    "override",
    [
        # Paired, because the example watches the learning rate and nothing would be recording it.
        "tracker=none callbacks=none",
        "tracker=csv",
        "tracker=clearml",
        "scheduler=onecycle",
        "scheduler=cosine",
        "scheduler=plateau",
        "scheduler=step",
        "callbacks=default",
        "model=dpt_dinov3",
        "model=unet",
        "trainer=profile",
        "loader=performance",
        "export=onnx",
        "export=pt2",
        "export=torchscript",
        "export=tensorrt",
        "export=ncnn",
        "export=all",
        # Paired with the encoder whose attention `lora.yaml` names; over a residual network it would
        # be refused at build, which is the point of the refusal rather than a fault of the group.
        "adapter=lora model=dpt_dinov3",
    ],
)
def test_every_group_option_validates_and_names_something_that_exists(override: str) -> None:
    """One entry may carry more than one group: some options only hold together in pairs."""
    config = load_config(composed("experiment=examples/classification", *override.split()))

    assert unresolved_names(config) == []


@pytest.mark.parametrize("example", EXAMPLES)
def test_every_name_a_shipped_example_writes_resolves_to_an_implementation(example: str) -> None:
    """Validating a file only proves its grammar: a name nothing implements passes and dies at the run."""
    config = load_config(composed(f"experiment=examples/{example}"))

    assert unresolved_names(config) == []


def tags_for(*overrides: str) -> list[str]:
    """The chips the shipped ClearML group writes for a run, blanks and all.

    Read before the tracker drops the blank ones, because the blank is the point: a chip names a
    section a run may not declare, and what tells "no chip" from "no run at all" is whether the group
    reached for it in a way that survives its absence.
    """
    return cast("list[str]", composed("tracker=clearml", *overrides)["tracker"]["tags"])


@pytest.mark.parametrize("example", EXAMPLES)
def test_a_shipped_example_can_be_filed_on_the_service_it_uploads_to(example: str) -> None:
    """A run declares `tracker: clearml` on top of whatever else it is, and every chip has to survive it.

    Between them the examples run with no scheduler, no delta, no teacher, a backbone written out by
    `_target_` and no picture at all — every absence a chip here is written around. One reaching a key
    such a run does not write ends the composition, so the run dies over a label it was never judged by.

    A chip is a value or nothing: `key=` written around a value a run may not have leaves the key
    standing with a hole after it, which is not blank and so is not dropped either.
    """
    tags = tags_for(f"experiment=examples/{example}")

    assert "adamw" in tags
    assert [tag for tag in tags if tag.endswith("=")] == []


def test_a_run_is_filed_under_what_makes_it_unlike_an_ordinary_one() -> None:
    """The sections an ordinary run leaves out are the ones worth filtering a list of runs by.

    Each is read from the declaration that already states it rather than written again: the family the
    heads are built over, the network itself under whichever key that family spells it with, the
    schedule, the delta learned beside weights held still, and the second network that teaches. A
    declaration is all a chip needs, so the run behind this one need not be one that would assemble.
    """
    ordinary = tags_for("experiment=examples/classification")
    unusual = tags_for(
        "experiment=examples/classification",
        "model=dpt_dinov3",
        "adapter=lora",
        "scheduler=cosine",
        "+learner={name: distillation}",
        "+model.neck={name: projector, width: 512}",
    )

    assert [tag for tag in ordinary if tag] == ["timm", "resnet18", "adamw", "lr=0.0003", "bs=32", "epochs=10"]
    assert set(unusual) - set(ordinary) == {
        "smp",
        "dpt",
        "tu-vit_small_plus_patch16_dinov3.lvd1689m",
        "cosine",
        "lora",
        "distillation",
        "projector",
    }


def test_a_component_written_out_by_import_path_leaves_a_blank_chip_rather_than_ending_the_run() -> None:
    """`_target_` writes no name anywhere in this grammar, and a chip reaching for one must not insist.

    A run whose optimizer arrives by import path is ordinary — every registry position takes either
    form — and the label it would be filed under is the last thing entitled to refuse it. Reaching
    straight for the name ends such a run before it is validated, naming the slot in a list of chips
    rather than the declaration or the fix.
    """
    tags = tags_for("experiment=examples/classification", "~optimizer.name", "+optimizer._target_=torch.optim.RAdam")

    assert "adamw" not in tags
    assert [tag for tag in tags if tag] == ["timm", "resnet18", "lr=0.0003", "bs=32", "epochs=10"]
