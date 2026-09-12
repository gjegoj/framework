"""Every so often, one batch becomes a page of samples wherever the run records to."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, override

import lightning as L

from src.callbacks.registry import callback_registry
from src.core import Stage
from src.integrations import Gallery, drawn_input
from src.tracking import ShowsPage
from src.training import StepPreview, TrainingData, TrainingModule
from src.visualization import MAX_DISPLAY_SIDE, HtmlRenderer

if TYPE_CHECKING:
    from src.core import Batch

log = logging.getLogger(__name__)


@callback_registry.register("samples")
class SampleGrid(L.Callback):
    """Draw a batch of samples, what is true against what was predicted, as one page in the tracker.

    The loop shows this callback every step it takes, and it draws the one it was asked for: a fixed
    batch, on a cadence, in the stages a run named. Everything else it is offered costs a comparison.

    What a run cannot draw is said once and never kills it: a page is a display, and a display that
    stops a fit is worse than no display. Both facts it needs are read off the run rather than declared
    a second time — the tasks from the module, and the statistics to undo from the input that applied
    them, so a run that changes its normalisation changes the page with it.

    Parameters:
        num_images: How many samples of the batch to draw.
        every_n_epochs: Draw on every Nth epoch *this stage runs* — see :meth:`_draw`.
        batch_index: Which batch of the epoch to draw — fixed, so drift between pages is visible.
        stages: Which stages draw; every stage by default. Training shows the pixels it trained on,
            augmentation and mixing included.
        title: What the page is called. The stage is appended, so a run's pages sort together and a
            tracker files each stage under its own name.
        max_side: Bound every inlined image and mask to this many pixels on its longest side;
            ``None`` inlines them whole, which a dense task turns into a very large page.
    """

    def __init__(
        self,
        num_images: int = 8,
        every_n_epochs: int = 5,
        batch_index: int = 0,
        # Read off the enum rather than listed again: a stage becomes drawable by existing.
        stages: Sequence[str] = tuple(Stage),
        title: str = "samples",
        *,
        max_side: int | None = MAX_DISPLAY_SIDE,
    ) -> None:
        super().__init__()
        _refuse_a_page_that_could_only_be_empty(
            num_images=num_images, every_n_epochs=every_n_epochs, batch_index=batch_index
        )
        self._stages = _declared_stages(stages)
        self._num_images = num_images
        self._every_n_epochs = every_n_epochs
        self._batch_index = batch_index
        self._title = title
        self._renderer = HtmlRenderer(max_side=max_side)
        self._gallery: Gallery | None = None
        self._trainer: L.Trainer | None = None
        self._reached: dict[Stage, int] = dict.fromkeys(self._stages, 0)
        self._said: set[str] = set()

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Work out what this run can be drawn from, and say what it cannot, before an epoch has run.

        The concrete module rather than a capability: a page needs both the run's tasks and a seam onto
        its steps, and one object declares both — asking for them separately would be two refusals for
        one missing thing.
        """
        self._trainer = trainer
        if not isinstance(pl_module, TrainingModule):
            self._say_once(
                "module",
                "The samples grid draws nothing: %s is not this framework's training module, so it holds "
                "no tasks to draw and no steps to be shown.",
                type(pl_module).__name__,
            )
            return
        self._gallery = self._gallery_for(trainer, pl_module)
        if self._gallery is None:
            return
        for name in self._gallery.undrawable:
            self._say_once(
                f"undrawable/{name}",
                "Task '%s' decides a number at every pixel, which no drawer has a shape for yet; it is "
                "left off the page and the run's other tasks are drawn.",
                name,
            )
        if not self._showing(trainer):
            self._say_once(
                "tracker",
                "No tracker this run records to can carry a page, so the samples grid will draw "
                "nothing. Both shipped trackers can — `clearml` renders it, `csv` writes it beside the "
                "numbers; `tracker: none` records nowhere. Drop the 'samples' callback to silence this.",
            )
        pl_module.preview_steps(self._draw)

    @override
    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Let go of the run when it is over.

        The module keeps the watcher — it offers no way to take one back, and one closure costs
        nothing. What it must not keep is a finished run: handing the same module to a second trainer
        is Lightning's own way of fitting and then testing, and a grid still holding the first would
        write that run's pages into the first run's tracker, under the first run's epoch number.
        """
        self._trainer = None

    def _gallery_for(self, trainer: L.Trainer, pl_module: TrainingModule) -> Gallery | None:
        """What this run's steps are drawn from, or nothing, said once with the reason."""
        data = trainer.datamodule  # type: ignore[attr-defined]
        if not isinstance(data, TrainingData):
            self._say_once(
                "data",
                "The samples grid draws nothing: %s is not the pipeline this framework prepares, so "
                "nothing here knows how an image was normalised.",
                type(data).__name__,
            )
            return None
        drawn = drawn_input(data.info.inputs)
        if drawn is None:
            self._say_once(
                "input",
                "The samples grid draws nothing: no input of this run declares the statistics it was "
                "normalised with, so no image of it can be shown as the file held it.",
            )
            return None
        return Gallery.of(pl_module.learner.tasks, *drawn)

    def _draw(self, preview: StepPreview) -> None:
        """Shown every step; this is the one it was asked for, or it is not.

        The cadence counts the epochs *this stage* runs rather than the trainer's epoch number. A run
        declaring ``trainer.check_val_every_n_epoch: 2`` validates only on odd epochs, and a cadence
        tested for divisibility against the epoch number then lands on none of them: measured over a
        twelve-epoch fit, that pairing drew no validation page at all for any even cadence. Counting
        opportunities also retires the special case the test stage needed — it runs once, and its one
        run is its first.

        Nothing that happens while drawing is allowed out of here. The loop calls this inside its own
        step, so a page that cannot be built — a mask at a resolution its label does not share, a log
        directory that is full, a tracker that cannot be reached — would otherwise end the fit at
        whichever epoch it first went wrong. A run losing its images is a smaller loss than a run.
        """
        trainer = self._trainer
        if self._gallery is None or trainer is None or not self._watched(trainer, preview):
            return
        self._reached[preview.stage] += 1
        if (self._reached[preview.stage] - 1) % self._every_n_epochs:
            return
        showing = self._showing(trainer)
        if not showing:
            return
        try:
            self._show(trainer, preview, showing)
        except Exception as error:
            self._say_once("drawing", "The samples grid stopped drawing after %s: %s", type(error).__name__, error)

    def _show(self, trainer: L.Trainer, preview: StepPreview, showing: list[ShowsPage]) -> None:
        assert self._gallery is not None
        title = f"{self._title}/{preview.stage}"
        views = self._gallery.views(preview.batch, preview.output, self._count(preview.batch))
        page = self._renderer.render(views, title=title, classes=self._gallery.classes)
        for tracker in showing:
            tracker.log_html(title, page, trainer.current_epoch)

    def _watched(self, trainer: L.Trainer, preview: StepPreview) -> bool:
        """Whether this step is one this grid watches at all — before any cadence is counted."""
        return (
            trainer.is_global_zero
            # A sanity check runs a validation batch before a single optimizer step. A page asking
            # where the model is wrong has no answer there, and it would land under the same title and
            # iteration as the first real epoch's — two artifacts, and no way to tell them apart.
            and not trainer.sanity_checking
            and preview.stage in self._stages
            and preview.batch_index == self._batch_index
        )

    def _count(self, batch: Batch) -> int:
        return min(self._num_images, len(batch))

    @staticmethod
    def _showing(trainer: L.Trainer) -> list[ShowsPage]:
        """Every backend that can carry a page.

        All of them rather than ``trainer.logger``, which is only the first: a run built from config
        declares one tracker, and a run wired by hand may hand the trainer several.
        """
        return [one for one in trainer.loggers if isinstance(one, ShowsPage)]

    def _say_once(self, topic: str, message: str, *args: Any) -> None:
        """Warn about a condition the first time it holds, then stay quiet.

        Lightning calls ``setup`` once per stage, so every one of these would otherwise repeat through
        a run — and a warning printed once per stage is one a reader stops seeing.
        """
        if topic in self._said:
            return
        self._said.add(topic)
        log.warning(message, *args)


def _declared_stages(stages: Sequence[str]) -> tuple[Stage, ...]:
    """Fail on a misspelt stage at build time rather than by drawing nothing for a whole run.

    ``Stage`` is the vocabulary, so it is asked rather than listed again here. Only the message is
    ours: it names every offending value at once, where ``Stage(value)`` would stop at the first.
    """
    unknown = [stage for stage in stages if stage not in set(Stage)]
    if unknown:
        raise ValueError(f"Unknown stage(s) for the samples grid: {', '.join(unknown)}. Valid: {', '.join(Stage)}.")
    return tuple(Stage(stage) for stage in stages)


def _refuse_a_page_that_could_only_be_empty(*, num_images: int, every_n_epochs: int, batch_index: int) -> None:
    """A value that can only draw nothing, named with the bound it broke.

    Keyword-only, because this list and the constructor's have to be kept in step by hand and three
    integers in a row is a transposition waiting to happen.
    """
    for name, value, lowest in (
        ("num_images", num_images, 1),
        ("every_n_epochs", every_n_epochs, 1),
        ("batch_index", batch_index, 0),
    ):
        if value < lowest:
            raise ValueError(f"The samples grid needs {name} >= {lowest}; got {value}.")
