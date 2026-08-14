"""The samples callback: due-gating, loud refusals, and one page per due batch."""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.callbacks.samples import SampleGrid
from src.config import ExperimentConfig
from src.core import Batch, Objective, Task, Topology
from src.core.entities import preview_of
from src.core.taxonomy import Stage
from src.losses import CrossEntropyCriterion
from src.models import CompositeModel, LinearHead, TaskComponents
from src.tasks.adapters import as_class_indices
from src.training import TrainingModule
from tests.support.fakes import Batches, FlattenBackbone, PageLogger
from tests.support.lightning import quiet_trainer
from tests.support.narrowing import tensor

MEAN = (0.0, 0.0, 0.0)
STD = (1.0, 1.0, 1.0)


def task_of() -> Task:
    return Task(
        name="label",
        topology=Topology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={},
        class_names=["cat", "dog"],
    )


def module() -> TrainingModule:
    return TrainingModule(
        model=CompositeModel(
            backbone=FlattenBackbone(dim=12),
            components={
                "label": TaskComponents(
                    head=LinearHead(12, 2),
                    criterion=CrossEntropyCriterion(),
                    activation=lambda logits: torch.softmax(logits, dim=1),
                    target_adapter=as_class_indices,
                )
            },
        ),
        tasks=[task_of()],
        optimizer_factory=partial(torch.optim.SGD, lr=0.1),
    )


def batch(**meta: Any) -> Batch:
    return Batch(
        inputs={"image": torch.rand(4, 3, 2, 2)},
        targets={"label": torch.tensor([0, 1, 0, 1])},
        meta={"cells": [{"image": f"images/{index}.png"} for index in range(4)], **meta},
    )


def grid(**overrides: Any) -> SampleGrid:
    declared: dict[str, Any] = {
        "tasks": [task_of()],  # a derived value the composition root offers
        "every_n_epochs": 1,
        "stages": ("val",),
        "mean": MEAN,
        "std": STD,
    }
    return SampleGrid(**(declared | overrides))


def drawn(callback: SampleGrid, trained: TrainingModule, logger: PageLogger, sample: Batch) -> PageLogger:
    """Drive the two calls Lightning makes: the step, then the hook it feeds the step's return to."""
    trainer = quiet_trainer(logger=logger)
    callback.setup(trainer, trained, stage="fit")
    outputs = trained.validation_step(sample, 0)
    callback.on_validation_batch_end(trainer, trained, outputs, sample, 0)
    return logger


def test_a_step_that_returns_no_preview_is_named_not_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a step can show what a step returns, so this is checked at the first drawn batch.

    The reference skipped it silently — one of six bare returns — and a run drew
    nothing while saying nothing.
    """
    trained = module()
    trainer = quiet_trainer(logger=(logger := PageLogger()))
    callback = grid()
    callback.setup(trainer, trained, stage="fit")

    with caplog.at_level(logging.WARNING):
        callback.on_validation_batch_end(trainer, trained, torch.tensor(1.0), batch(), 0)
        callback.on_validation_batch_end(trainer, trained, torch.tensor(1.0), batch(), 0)

    assert logger.pages == []
    said = [record.message for record in caplog.records if "StepPreview" in record.message]
    assert len(said) == 1  # once, however many batches pass
    assert "Tensor" in said[0]


def test_a_task_the_preview_does_not_carry_is_named_not_silently_lost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A module that previews only some of the declared tasks loses the rest from the
    page; the loss is said once, naming the task and what the preview lacked, and the
    tasks the preview does carry are still drawn."""
    extra = Task(
        name="extra",
        topology=Topology.GLOBAL,
        objective=Objective.MULTICLASS,
        metrics={},
        class_names=["a", "b"],
    )
    callback = grid(tasks=[task_of(), extra])

    with caplog.at_level(logging.WARNING):
        logger = drawn(callback, module(), PageLogger(), batch())
        drawn(callback, module(), logger, batch())

    assert len(logger.pages) == 2  # the page itself survives the missing task
    said = [record.message for record in caplog.records if "'extra'" in record.message]
    assert len(said) == 1  # once, however many batches pass
    assert "outputs and targets" in said[0]


def test_a_tracker_without_log_html_warns_once_and_the_run_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A CSV run with the callback declared should say so once, not die and not stay silent."""
    trainer = quiet_trainer()
    callback = grid()
    trained = module()

    with caplog.at_level(logging.WARNING):
        callback.setup(trainer, trained, stage="fit")
        callback.setup(trainer, trained, stage="test")  # Lightning calls setup per stage

    assert len([record for record in caplog.records if "log_html" in record.message]) == 1


def test_the_grid_denormalises_by_what_the_root_config_normalises_by() -> None:
    """Two readers of one set of numbers, and a drift between them is invisible.

    The transforms normalise by the root's `mean`/`std`; this grid undoes exactly
    that to show the pixels back. If the two defaults were written separately and
    one moved, every page would be mis-coloured — and wrong colours read as a model
    problem, which is the one thing a page of samples exists to rule out.
    """
    shipped = SampleGrid(tasks=[task_of()])
    root = ExperimentConfig.model_fields["mean"].get_default(call_default_factory=True)

    assert shipped._mean.flatten().tolist() == pytest.approx(root)


def test_the_default_draws_on_every_stage_the_framework_has() -> None:
    """Derived from `Stage`, so a new member is drawable by existing rather than by an edit here."""
    assert set(SampleGrid(tasks=[task_of()])._stages) == set(Stage)


def test_an_unknown_stage_is_refused_with_the_valid_ones() -> None:
    """At assembly, not by drawing nothing for a whole run."""
    with pytest.raises(ValueError, match="train, val, test"):
        grid(stages=("validation",))


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ({"num_images": 0}, "num_images >= 1"),
        ({"every_n_epochs": 0}, "every_n_epochs >= 1"),
        ({"batch_index": -1}, "batch_index >= 0"),
        ({"threshold": 1.5}, r"\[0, 1\]"),
        ({"std": (1.0,)}, "one std per mean"),
        # `max_side: 0` scales every picture and every mask to a single pixel, and
        # the page builds and uploads a grid of dots without a word.
        ({"max_side": 0}, "max_side >= 1"),
        ({"max_chip_chars": 0}, "max_chip_chars >= 1"),
    ],
)
def test_a_value_that_could_only_draw_nothing_is_refused_by_name(declared: dict[str, Any], expected: str) -> None:
    """Assembly is where these fail; an epoch of silence is not a diagnosis."""
    with pytest.raises(ValueError, match=expected):
        grid(**declared)


def test_the_page_is_drawn_from_the_hooks_own_argument_not_from_anything_kept() -> None:
    """A step returns its preview and Lightning hands it back per batch; nothing is retained.

    Written so it can fail. Two batches with opposite truth go through the hook in
    turn, and each page must show its own. Anything retained between calls — a
    module field, a cache in the callback — leaves the second page labelled with the
    first batch's truth, which is the whole failure mode returning-not-remembering
    removes. The earlier version of this test passed ``outputs=None``, so ``_draw``
    bailed before it reached the batch and the assertion held under any
    implementation, including one that drew the wrong batch.
    """
    trained, logger = module(), PageLogger()
    trainer = quiet_trainer(logger=logger)
    callback = grid()
    callback.setup(trainer, trained, stage="fit")
    cats, dogs = batch(), batch()
    cats.targets["label"] = torch.zeros(4, dtype=torch.long)
    dogs.targets["label"] = torch.ones(4, dtype=torch.long)

    callback.on_validation_batch_end(trainer, trained, trained.validation_step(cats, 0), cats, 0)
    callback.on_validation_batch_end(trainer, trained, trained.validation_step(dogs, 0), dogs, 0)

    first, second = logger.pages[0][1], logger.pages[1][1]
    assert 'data-key="label::gt::cat"' in first and 'data-key="label::gt::dog"' not in first
    assert 'data-key="label::gt::dog"' in second and 'data-key="label::gt::cat"' not in second


def test_the_grid_asks_for_a_preview_only_on_the_batch_it_will_draw() -> None:
    """One policy, read twice: what it declares before the step is what it draws after.

    The request is what keeps a preview from being built on every step of every run
    — it shares storage with the activated outputs, which Lightning then holds alive
    through the optimizer step. If this drifted from `_is_due`, the grid would
    either draw nothing or pay for every batch again.
    """
    callback = grid(batch_index=1)
    trainer = quiet_trainer(logger=PageLogger())
    trained = module()

    asked = []
    for index in range(3):
        callback.on_validation_batch_start(trainer, trained, batch(), index)
        asked.append(callback.awaiting_preview)

    assert asked == [False, True, False]


def test_a_due_validation_batch_becomes_one_page() -> None:
    logger = drawn(grid(), module(), PageLogger(), batch())

    title, page, iteration = logger.pages[0]
    assert title == "samples/val"
    assert "label::gt::" in page
    assert "label::pred::" in page
    assert iteration == 0


def test_the_page_draws_the_forward_that_trained_not_a_second_one() -> None:
    """Same weights, same batch: the chips must agree with the step the module kept."""
    trained = module()
    sample = batch()
    logger = drawn(grid(), trained, PageLogger(), sample)

    kept = preview_of(trained.validation_step(sample, 0))
    assert kept is not None
    predicted = int(tensor(kept.outputs["label"])[0].argmax().item())
    assert f'data-key="label::pred::{["cat", "dog"][predicted]}"' in logger.pages[0][1]


def test_the_readable_cell_reaches_the_page_as_the_samples_source() -> None:
    """The provenance path end to end: a table cell becomes the pill on a cell."""
    logger = drawn(grid(), module(), PageLogger(), batch())

    assert 'data-copy="images/0.png"' in logger.pages[0][1]


def test_a_cell_with_no_picture_behind_it_is_drawn_as_text() -> None:
    """A caption reaches the model as input_ids; the words survive only in the row."""
    sample = batch(cells=[{"image": f"images/{i}.png", "caption": "a cat"} for i in range(4)])
    sample.inputs["caption"] = torch.zeros(4, 6, dtype=torch.long)

    logger = drawn(grid(), module(), PageLogger(), sample)

    assert "a cat" in logger.pages[0][1]


def test_an_undeclared_stage_is_not_drawn() -> None:
    logger = drawn(grid(stages=("test",)), module(), PageLogger(), batch())

    assert logger.pages == []


def test_a_batch_that_is_not_the_chosen_one_is_not_drawn() -> None:
    """A fixed batch index keeps the same samples on the page as epochs pass."""
    trained = module()
    sample = batch()
    trainer = quiet_trainer(logger=(logger := PageLogger()))
    callback = grid()
    callback.setup(trainer, trained, stage="fit")
    outputs = trained.validation_step(sample, 0)

    callback.on_validation_batch_end(trainer, trained, outputs, sample, 3)

    assert logger.pages == []


def fitted(callback: SampleGrid, root: Path, **trainer_kwargs: Any) -> PageLogger:
    """A real fit, because the defects this pins only appear when Lightning drives."""
    logger = PageLogger()
    loader = DataLoader(Batches([batch(), batch()]), batch_size=None)
    trainer = quiet_trainer(
        logger=logger,
        callbacks=[callback],
        default_root_dir=root,
        log_every_n_steps=1,
        **trainer_kwargs,
    )
    trainer.fit(module(), train_dataloaders=loader, val_dataloaders=loader)
    return logger


def test_the_sanity_check_does_not_produce_a_page(tmp_path: Path) -> None:
    """It runs a val batch before a single optimizer step, and lands on the same iteration.

    Two artifacts under one title and one iteration: a tracker shows one of them
    and nothing says which. The page's question has no answer at step zero anyway.
    """
    logger = fitted(grid(), tmp_path, num_sanity_val_steps=2)

    assert [(title, iteration) for title, _, iteration in logger.pages] == [("samples/val", 0)]


def test_a_page_is_drawn_from_the_batch_the_hook_was_handed(tmp_path: Path) -> None:
    """End to end through Lightning: the preview a callback reads is its own batch's."""
    logger = fitted(grid(), tmp_path, num_sanity_val_steps=0)

    assert len(logger.pages) == 1
    assert "label::pred::" in logger.pages[0][1]


def test_a_test_pass_draws_whatever_the_fits_epoch_count_happened_to_be(tmp_path: Path) -> None:
    """A test pass runs once and has no epochs, so an epoch cadence cannot gate it.

    Lightning reports ``current_epoch`` during test as the fit's final count, and
    ``run`` drives fit and test on one trainer — so ``epochs % every_n_epochs``
    silently decided whether a test grid existed at all. Measured: ``max_epochs=3``
    with the default cadence of 5 drew nothing and said nothing. Three is chosen
    here for exactly that reason.
    """
    logger, trained = PageLogger(), module()
    loader = DataLoader(Batches([batch(), batch()]), batch_size=None)
    trainer = quiet_trainer(
        logger=logger,
        max_epochs=3,
        callbacks=[grid(stages=("test",), every_n_epochs=5)],
        default_root_dir=tmp_path,
        log_every_n_steps=1,
    )

    trainer.fit(trained, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(trained, dataloaders=loader, verbose=False)

    assert [title for title, _, _ in logger.pages] == ["samples/test"]


def test_the_drawn_pixels_are_the_source_image_back_again() -> None:
    """Checking a normalisation or a colour-space mistake is the one job mean/std are here for.

    ``.byte()`` truncates toward zero rather than rounding, which measured a grey
    level off the source on 62 of 256 values — so a pixel diff against the original
    never came back clean, and the check the page exists to enable did not work.
    """
    ramp = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16) / 255.0
    mean, std = (0.485,), (0.229,)
    callback = grid(mean=mean, std=std)

    drawn_pixels = callback._to_uint8((ramp - mean[0]) / std[0], count=1)

    assert np.array_equal(drawn_pixels[0, ..., 0], (ramp[0, 0] * 255).round().to(torch.uint8).numpy())


def test_an_input_with_more_channels_than_mean_values_is_skipped_and_named(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 4-band input against a 3-value mean used to die inside the denormalisation.

    A bare shape mismatch, an hour into a run, naming neither the callback nor the
    input — and raised from a batch-end hook, so it took the run with it over a
    picture. It is skipped and named instead, and the rest of the page still draws.
    """
    sample = batch()
    sample.inputs["multispectral"] = torch.rand(4, 5, 2, 2)

    with caplog.at_level(logging.WARNING):
        logger = drawn(grid(), module(), PageLogger(), sample)

    assert logger.pages  # the run goes on, and the ordinary picture is still drawn
    said = [record.message for record in caplog.records if "multispectral" in record.message]
    assert len(said) == 1
    assert "5 channels" in said[0] and "3 mean/std" in said[0]
