"""Every N epochs, one batch becomes a self-contained HTML grid in the tracker."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

import lightning as L
import torch

# At runtime, not under TYPE_CHECKING: a page asks what shape a task produced before
# it looks for an annotator, and that question is answered by an isinstance.
from src.core.entities import Instances, StepPreview, preview_of
from src.core.normalisation import IMAGENET_MEAN, IMAGENET_STD
from src.core.reporting import HtmlLogger
from src.core.taxonomy import Stage
from src.visualization import (
    MAX_CHIP_CHARS,
    MAX_DISPLAY_SIDE,
    HtmlRenderer,
    Image,
    SampleView,
    Text,
    build_annotators,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from torch import Tensor

    from src.core.entities import Batch, Task
    from src.visualization import Annotator, Media

log = logging.getLogger(__name__)


class SampleGrid(L.Callback):
    """Draw a batch of samples, ground truth against prediction, as one HTML page.

    A step returns its preview to this callback's batch-end hook beside the batch it came
    from, built only for batches this callback asked for in ``on_*_batch_start``
    (``AwaitsPreview``). What a run will not draw is said early, and never kills a run.

    Parameters:
        tasks (Sequence[Task]): Every task whose predictions may be drawn; offered by assembly.
        mean (Sequence[float]): The normalisation mean the transforms applied; ``mean: "${mean}"``.
        std (Sequence[float]): The matching standard deviation.
        num_images (int): How many samples of the batch to draw.
        every_n_epochs (int): Draw on fit epochs divisible by this; a test pass always draws.
        batch_index (int): Which batch of the epoch to draw — fixed, so drift is visible.
        stages (Sequence[str]): Which stages draw; every stage by default. Train shows augmented pixels.
        title (str): The page's title, and the tracker series it lands under.
        threshold (float): Offered to the annotators that name it (binary, multilabel).
        ignore_index (int | None): Offered to the dense annotator that names it.
        max_side (int | None): Downscale pictures to fit this before inlining; ``None`` inlines whole.
        max_chip_chars (int): Chip text budget before truncation; the lightbox shows the full text.
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        num_images: int = 8,
        every_n_epochs: int = 5,
        batch_index: int = 0,
        # Derived from the enum rather than listed: a new stage becomes drawable by
        # existing, and there is no private copy to fall out of step with it.
        stages: Sequence[str] = tuple(Stage),
        title: str = "samples",
        threshold: float = 0.5,
        ignore_index: int | None = None,
        max_side: int | None = MAX_DISPLAY_SIDE,
        max_chip_chars: int = MAX_CHIP_CHARS,
    ) -> None:
        super().__init__()
        _refuse_impossible_values(
            mean=mean,
            std=std,
            num_images=num_images,
            every_n_epochs=every_n_epochs,
            batch_index=batch_index,
            threshold=threshold,
        )
        self._stages = _valid_stages(stages)
        self._mean = torch.tensor(list(mean)).view(1, -1, 1, 1)
        self._std = torch.tensor(list(std)).view(1, -1, 1, 1)
        self._num_images = num_images
        self._every_n_epochs = every_n_epochs
        self._batch_index = batch_index
        self._title = title
        self._renderer = HtmlRenderer(max_chip_chars=max_chip_chars, max_side=max_side)
        self._tasks = tuple(tasks)
        # Built here, not at setup: the tasks are known at assembly, so a task that
        # draws nothing says so before the run starts rather than after an epoch.
        self._annotators = build_annotators(self._tasks, threshold=threshold, ignore_index=ignore_index)
        self._said: set[str] = set()
        self._awaiting = False

    @property
    def awaiting_preview(self) -> bool:
        """Whether the step about to run is one this grid will draw — the ``AwaitsPreview`` port.

        Answered from the batch-start hooks below, which Lightning runs before the
        step, so a preview is built only for the batches that become a page. The
        answer is the same ``_is_due`` the drawing side asks: one policy, in one
        place, read twice.
        """
        return self._awaiting

    @override
    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Batch, batch_idx: int
    ) -> None:
        self._awaiting = self._is_due(trainer, Stage.TRAIN, batch_idx)

    @override
    def on_validation_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._awaiting = self._is_due(trainer, Stage.VAL, batch_idx)

    @override
    def on_test_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._awaiting = self._is_due(trainer, Stage.TEST, batch_idx)

    @override
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Warn about a tracker that cannot show a page — once, however many stages run."""
        if not self._page_targets(trainer):
            self._say_once(
                "tracker",
                "No configured logger implements log_html, so the samples grid will draw nothing. "
                "The clearml logger provides it; drop the 'samples' callback to silence this.",
            )

    @override
    def on_train_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Batch, batch_idx: int
    ) -> None:
        self._draw(trainer, Stage.TRAIN, outputs, batch, batch_idx)

    @override
    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._draw(trainer, Stage.VAL, outputs, batch, batch_idx)

    @override
    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._draw(trainer, Stage.TEST, outputs, batch, batch_idx)

    def _draw(self, trainer: L.Trainer, stage: Stage, outputs: Any, batch: Batch, index: int) -> None:
        if not self._is_due(trainer, stage, index):
            return
        targets = self._page_targets(trainer)
        if not targets:
            return
        preview = preview_of(outputs)
        if preview is None:
            self._say_once(
                "step",
                # The one thing that cannot be checked at assembly: only a step can show
                # what a step returns. Named at the first batch that would have been
                # drawn, so a run never loses its pages in silence.
                "The samples grid draws nothing: this module's step returned %s, with no StepPreview "
                "under '%s'. TrainingModule returns one; a module of your own must too.",
                type(outputs).__name__,
                StepPreview.KEY,
            )
            return
        views = self._views(batch, preview)
        if not views:
            return
        title = f"{self._title}/{stage}"
        page = self._renderer.render(views, title=title, classes=self._class_names())
        for logger in targets:
            logger.log_html(title=title, html=page, iteration=trainer.current_epoch)

    def _class_names(self) -> dict[str, Sequence[str]]:
        """Each task's whole vocabulary, so a class keeps its colour from page to page."""
        return {task.name: task.class_names for task in self._tasks if task.class_names}

    def _is_due(self, trainer: L.Trainer, stage: Stage, index: int) -> bool:
        return (
            trainer.is_global_zero
            # Lightning's sanity check runs a val batch before a single optimizer
            # step. A page asking where the model is wrong has no answer there, and
            # it would land under the same title and iteration as the first real
            # epoch's — two artifacts, and no way to tell which one a tracker shows.
            and not trainer.sanity_checking
            and stage in self._stages
            and index == self._batch_index
            and self._epoch_is_due(trainer, stage)
        )

    def _epoch_is_due(self, trainer: L.Trainer, stage: Stage) -> bool:
        """A cadence in epochs, for the stages that have epochs.

        A test pass runs once, and Lightning reports ``current_epoch`` there as the
        fit's final count — so gating it on a cadence made whether a test grid was
        drawn at all depend on ``max_epochs`` being divisible by an unrelated knob.
        Measured: after ``fit(max_epochs=12)`` with the default cadence of 5, a run
        declaring ``stages: [test]`` drew nothing and said nothing.
        """
        return stage is Stage.TEST or trainer.current_epoch % self._every_n_epochs == 0

    @staticmethod
    def _page_targets(trainer: L.Trainer) -> list[HtmlLogger]:
        """Every backend that can carry a page, in the shape every artifact consumer uses.

        Silent when none can: the warn-once in ``setup`` has already said so, at the one
        moment a user can act on it, and naming it again per epoch would be noise.
        """
        return [one for one in trainer.loggers if isinstance(one, HtmlLogger)]

    def _views(self, batch: Batch, preview: StepPreview) -> list[SampleView]:
        drawable = {alias: tensor for alias, tensor in batch.inputs.items() if self._is_drawable(alias, tensor)}
        self._warn_once_about_shared_normalisation(drawable)
        cells = batch.cells
        # Decided before denormalising, so one number governs the page and the
        # conversion does exactly the work the page uses.
        count = self._count(batch, drawable)
        pictures = {alias: self._to_uint8(tensor, count) for alias, tensor in drawable.items()}
        drawn = self._drawn(preview)
        views: list[SampleView] = []
        for index in range(count):
            row = cells[index] if index < len(cells) else {}
            view = SampleView(media=_media(batch, index, pictures, row))
            for task, annotator, outputs, targets in drawn:
                annotator.annotate(view, task, outputs, targets, index)
            views.append(view)
        return views

    def _drawn(self, preview: StepPreview) -> list[tuple[Task, Annotator, Tensor, Tensor]]:
        """The tasks this page will annotate, resolved once — these are facts about a
        task, not about a sample, so they are read before the per-sample loop."""
        drawn: list[tuple[Task, Annotator, Tensor, Tensor]] = []
        for task in self._tasks:
            annotator = self._annotators.get(task.name)
            if annotator is None:
                # Already named at assembly: build_annotators logs the task and the reason.
                continue
            outputs = preview.outputs.get(task.name)
            targets = preview.targets.get(task.name)
            if outputs is None or targets is None:
                missing = " and ".join(
                    name for name, value in (("outputs", outputs), ("targets", targets)) if value is None
                )
                self._say_once(
                    f"missing/{task.name}",
                    # The one silent loss the assembly checks could not catch: only a step
                    # shows which tasks its preview carries. Named at the first drawn batch,
                    # while the tasks the preview does carry still make the page.
                    "Task '%s' has an annotator, but the step's preview carries no %s for it, "
                    "so it is left off the page. TrainingModule previews every task; "
                    "a module of your own must too.",
                    task.name,
                    missing,
                )
                continue
            if isinstance(outputs, Instances) or isinstance(targets, Instances):
                # A page draws what it can and says what it cannot: a task predicting a
                # set of objects has no annotator yet, and a run should not lose the
                # tasks that do have one over the task that does not.
                self._say_once(
                    f"undrawable/{task.name}",
                    "Task '%s' predicts a set of objects, which no annotator draws yet; "
                    "it is left off the page and the other tasks are drawn.",
                    task.name,
                )
                continue
            drawn.append((task, annotator, outputs, targets))
        return drawn

    def _say_once(self, topic: str, message: str, *args: Any) -> None:
        """Warn about a condition the first time it holds, and then stay quiet.

        Lightning calls ``setup`` once per stage and the batch hooks once per batch,
        so every one of these would otherwise repeat for a whole run — and a warning
        printed 300 times is one nobody reads. Keyed by topic rather than by a flag
        per condition, so a new thing to warn about needs no new state.
        """
        if topic in self._said:
            return
        self._said.add(topic)
        log.warning(message, *args)

    def _is_drawable(self, alias: str, tensor: Any) -> bool:
        """A picture, and one whose channels this callback's mean and std can undo.

        Fewer channels than declared is the grayscale case and slices cleanly; more (a 4-band
        input against a 3-value mean) is skipped and named instead of dying in denormalisation.
        """
        if not _is_picture(tensor):
            return False
        channels, declared = int(tensor.shape[1]), int(self._mean.shape[1])
        if channels > declared:
            self._say_once(
                f"channels:{alias}",
                "The samples grid skips input '%s': it has %d channels and only %d mean/std "
                "value(s) were given, so its normalisation cannot be undone.",
                alias,
                channels,
                declared,
            )
            return False
        return True

    def _warn_once_about_shared_normalisation(self, drawable: dict[str, Tensor]) -> None:
        """One mean and one std cannot undo two different normalisations.

        A run with per-input transforms would have its second picture drawn in the
        wrong colours, and wrong colours look like a model problem. Said out loud
        rather than guessed at; a per-input presentation contract is in the backlog.
        """
        if len(drawable) > 1:
            self._say_once(
                "colours",
                "The samples grid draws %s with one mean/std pair. If these inputs were "
                "normalised differently, every picture but the first will be mis-coloured.",
                ", ".join(sorted(drawable)),
            )

    def _count(self, batch: Batch, drawable: dict[str, Tensor]) -> int:
        """How many samples to draw; loud when the batch shows nothing drawable at all."""
        sizes = [int(tensor.shape[0]) for tensor in drawable.values()]
        rows = batch.cells
        if not sizes and not rows:
            shapes = {alias: getattr(tensor, "shape", None) for alias, tensor in batch.inputs.items()}
            log.warning(
                "The samples grid found nothing to draw: no [B, C, H, W] float input and no readable "
                "cells in the batch metadata. Inputs: %s.",
                shapes,
            )
            return 0
        return min([self._num_images, *sizes]) if sizes else min(self._num_images, len(rows))

    def _to_uint8(self, tensor: Tensor, count: int) -> np.ndarray:
        """Undo the run's own normalisation and lay the channels out the way a browser reads them."""
        images = tensor[:count].detach().cpu().float()
        channels = images.shape[1]
        mean = self._mean[:, :channels]
        std = self._std[:, :channels]
        # Rounded, not truncated: `.byte()` cuts toward zero, which measured a level
        # off the source on 62 of 256 values — and checking a normalisation against
        # the original image is the whole job the mean/std knobs are here for.
        images = (images * std + mean).clamp(0.0, 1.0).mul(255).round().byte()
        if channels == 1:
            images = images.repeat(1, 3, 1, 1)
        pixels: np.ndarray = images.permute(0, 2, 3, 1).numpy()
        return pixels


def _is_picture(tensor: Any) -> bool:
    """The shape rule, read off the tensor: a ``[B, C, H, W]`` float input is a picture."""
    return isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.is_floating_point()


def _media(batch: Batch, index: int, pictures: dict[str, np.ndarray], row: dict[str, str]) -> dict[str, Media]:
    """Every input this sample can show, in the order the batch declares them.

    A picture draws itself; anything else draws its readable cell, because a text
    input reaches the model as ``input_ids`` and no tokenizer lives here. An input
    that is neither draws nothing rather than a placeholder.
    """
    media: dict[str, Media] = {}
    for alias in batch.inputs:
        if alias in pictures:
            media[alias] = Image(pixels=pictures[alias][index], source=row.get(alias))
        elif alias in row:
            media[alias] = Text(text=row[alias])
    return media


def _valid_stages(stages: Sequence[str]) -> tuple[Stage, ...]:
    """Fail at assembly on a misspelt stage rather than by drawing nothing for a whole run.

    ``Stage`` is the vocabulary, so it is asked rather than re-listed here — a new
    member becomes drawable by existing. Only the message is ours: it names every
    offending value at once, where ``Stage(value)`` would stop at the first.
    """
    unknown = [stage for stage in stages if stage not in set(Stage)]
    if unknown:
        raise ValueError(f"Unknown stage(s) for the samples grid: {', '.join(unknown)}. Valid: {', '.join(Stage)}.")
    return tuple(Stage(stage) for stage in stages)


def _refuse_impossible_values(
    *,
    mean: Sequence[float],
    std: Sequence[float],
    num_images: int,
    every_n_epochs: int,
    batch_index: int,
    threshold: float,
) -> None:
    """Fail at assembly on a value that can only draw nothing, naming it and the bound.

    Keyword-only, because this list and the constructor's have to stay in step by hand
    and eight positionals in that order was a transposition waiting to happen.

    The page's own bounds — ``max_side``, ``max_chip_chars`` — are not here: they belong
    to ``HtmlRenderer``, which is public and was reachable without them being checked at
    all. A knob is refused by whoever owns it.
    """
    if len(mean) != len(std):
        raise ValueError(
            f"The samples grid needs one std per mean: got {len(mean)} mean value(s) and {len(std)} std value(s)."
        )
    for name, value, lowest in (
        ("num_images", num_images, 1),
        ("every_n_epochs", every_n_epochs, 1),
        ("batch_index", batch_index, 0),
    ):
        if value < lowest:
            raise ValueError(f"The samples grid needs {name} >= {lowest}; got {value}.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"The samples grid's threshold compares against probabilities, so it must lie in [0, 1]; got {threshold}."
        )
