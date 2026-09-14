"""Familiar experiment controls, independent of the model's input modalities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schema import ComponentConfig, ModelConfig, PreprocessingConfig, TaskConfig
from src.core import Stage, validate_name

LEARNING_RATE_MONITOR = "lr_monitor"
"""The shipped callback that only reports, named here because the pairing is a rule about two sections."""

FREEZE = "freeze"
"""The shipped callback that holds parameters still, named here for the same reason as the one above."""

DISTILLATION = "distillation"
"""The shipped learner that reads a second network, named here for the same reason again.

Spelled rather than imported: this package reads no other, so a registry name reaches it as a word. The
price is the one the two above already pay — a component declared by ``_target_`` writes no name, and the
pairing below then holds for nobody.
"""


def refuse_a_path_that_is_not_there(key: str, path: str | None) -> None:
    """A declared file is answered for where it is declared, not minutes later where something opens it.

    One home because two sections name one: the weights a run starts from, and the weights a teacher
    answers with. Both would otherwise be found missing after the sources were read and the cache warmed.
    """
    if path is not None and not Path(path).is_file():
        raise ValueError(f"{key} names no file: {path}")


def _reaches(frozen: str, adapted: str) -> bool:
    """Whether holding one dot-path still holds the other: the same module, or either one inside the other."""
    return frozen == adapted or frozen.startswith(f"{adapted}.") or adapted.startswith(f"{frozen}.")


def refuse_owned_keys(values: Mapping[str, object], owned: Mapping[str, str]) -> None:
    """A key the framework settles elsewhere is declared once, where it belongs."""
    for key, location in owned.items():
        if key in values:
            raise ValueError(f"Declare {key} once, at {location}.")


def refuse_settled_keys(values: Mapping[str, object], settled: Mapping[str, str]) -> None:
    """A key this framework never takes a declaration for, and why — there is nowhere to move it to."""
    for key, reason in settled.items():
        if key in values:
            raise ValueError(f"{key} is not a declaration this framework takes: {reason}.")


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = None
    name: str | None = None
    train: bool = True
    test: bool = True
    checkpoint_path: str | None = Field(
        None, min_length=1, description="Weights to start from, or the run being reported on where none are learned."
    )
    resume_path: str | None = Field(None, min_length=1, description="Continue weights, optimizer, scheduler and epoch.")
    directory: str = Field("runs", min_length=1)

    @property
    def scores_without_training(self) -> bool:
        """Whether this run will report numbers about a model it has no chance to learn anything about.

        Derived rather than declared: a scoring stage is the only thing that reports, and a run that
        does not train can only have got what it reports on out of a file. Read where a checkpoint is
        opened, because that is where the difference between a starting point and the subject of a
        report has to be made.
        """
        return self.test and not self.train

    @model_validator(mode="after")
    def restoration(self) -> RunConfig:
        if sum(path is not None for path in (self.checkpoint_path, self.resume_path)) > 1:
            raise ValueError("Choose one of run.checkpoint_path or run.resume_path.")
        if self.resume_path is not None and not self.train:
            raise ValueError("run.resume_path requires run.train=true.")
        for field, path in (("checkpoint_path", self.checkpoint_path), ("resume_path", self.resume_path)):
            refuse_a_path_that_is_not_there(f"run.{field}", path)
        return self


class ForwardSection(BaseModel):
    """Declared fields are validated; every other key reaches the library constructor verbatim.

    Keys the framework owns elsewhere are refused by name, so a value is declared once.
    """

    model_config = ConfigDict(extra="allow")
    owned: ClassVar[Mapping[str, str]] = {}
    """Keys declared once somewhere else in this file, by the path a reader can go and edit."""
    settled: ClassVar[Mapping[str, str]] = {}
    """Keys with no declaration anywhere: the framework decides them, and says why."""

    @model_validator(mode="after")
    def framework_arguments(self) -> Self:
        refuse_owned_keys(self.params, self.owned)
        refuse_settled_keys(self.params, self.settled)
        return self

    @property
    def params(self) -> dict[str, object]:
        """The forwarded keys alone; declared fields are read as attributes."""
        return dict(self.model_extra or {})


class LoaderConfig(ForwardSection):
    owned: ClassVar[Mapping[str, str]] = {
        "batch_size": "the root's batch_size",
        "dataset": "data",
        "collate_fn": "preprocessing.collator",
    }
    settled: ClassVar[Mapping[str, str]] = {
        "shuffle": "training shuffles and evaluation preserves the order it was given",
        "batch_sampler": "it replaces the fixed batch size, so a run that needs one declares a data "
        "module that yields batches",
    }

    num_workers: int = Field(0, ge=0)
    pin_memory: bool = False
    drop_last: bool = Field(False, description="Training only; evaluation preserves every sample.")


class TrainerConfig(ForwardSection):
    owned: ClassVar[Mapping[str, str]] = {
        "max_epochs": "the root's epochs",
        "logger": "the root's tracker",
        "callbacks": "the root's callbacks",
        "default_root_dir": "run.directory",
    }

    accelerator: str = "auto"
    devices: int | str | list[int] = "auto"
    profiler: ComponentConfig | None = None


class SchedulerConfig(ComponentConfig):
    interval: Literal["epoch", "step"] = "epoch"
    frequency: int = Field(1, gt=0)
    monitor: str | None = None
    strict: bool = True


class TeacherConfig(ModelConfig):
    """A second network for a run to learn from, and the weights that make it worth learning from.

    An ordinary model declaration, because that is what it is: the same grammar, built by the same
    builder, sized by the same tasks. Its heads are not written here for exactly that reason — a teacher
    answers the questions this run asks, and a second statement of their shapes could disagree.

    The weights are not optional, and that is the whole of what this section adds. A network whose head
    was only just initialised answers with noise, and a run distilling from it would descend towards
    nothing while every number it reports looks ordinary. The file is one this framework wrote, since
    that is what a run's own checkpoint is; weights in someone else's shape are a phase of their own.
    """

    checkpoint_path: str = Field(min_length=1, description="The run whose weights this teacher answers with.")

    @model_validator(mode="after")
    def taught(self) -> TeacherConfig:
        refuse_a_path_that_is_not_there("teacher.checkpoint_path", self.checkpoint_path)
        return self


class ExperimentConfig(BaseModel):
    """Assembly injects root controls; runtime objects never read this schema.

    Preprocessing owns modality-specific loading, normalization and collation. Its component may
    compose named inputs or wrap one joint image/text processor, and every run declares one: a run
    over inputs that are already tensors arrives with the data module that yields them.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    lr: float = Field(1e-3, gt=0, allow_inf_nan=False)
    batch_size: int = Field(16, gt=0, strict=True)
    epochs: int = Field(10, gt=0, strict=True)
    data: ComponentConfig
    preprocessing: PreprocessingConfig
    transforms: dict[Stage, ComponentConfig] = Field(
        default_factory=dict, description="Stage-specific sample augmentation, before final normalization and encoding."
    )
    model: ModelConfig
    tasks: dict[str, TaskConfig]
    learner: ComponentConfig = Field(default_factory=lambda: ComponentConfig(name="standard"))
    optimizer: ComponentConfig = Field(default_factory=lambda: ComponentConfig(name="adamw"))
    scheduler: SchedulerConfig | None = None
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    callbacks: list[ComponentConfig] = Field(default_factory=list)
    teacher: TeacherConfig | None = None
    adapter: ComponentConfig | None = Field(
        None, description="Parameters added to a named part of the model before training and folded back after."
    )
    tracker: ComponentConfig | None = None
    export: list[ComponentConfig] = Field(
        default_factory=list, description="Deployment formats the trained model is written in when the run ends."
    )
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def connections(self) -> ExperimentConfig:
        """Everything two sections have to agree about, checked before anything is built from them.

        Here rather than in the composition root: these are functions of the declaration alone, and a
        declaration that has been validated should be one a run can be built from. Checking them at the
        root would leave a second reader — a notebook, a test, an export entry point — holding a config
        that passed validation and still cannot be assembled, and would refuse a typo only after the
        global seed had already been set.
        """
        refuse_owned_keys(self.optimizer.params, {"lr": "the root's lr"})
        if not self.tasks:
            raise ValueError("An experiment requires at least one task.")
        for name in self.tasks:
            validate_name(name, label="Task")
        self._refuse_heads_declared_twice()
        self._refuse_inputs_that_disagree()
        self._refuse_watching_a_rate_with_nothing_recording()
        self._refuse_freezing_what_this_run_adapts()
        self._refuse_half_of_a_distillation()
        return self

    def _refuse_half_of_a_distillation(self) -> None:
        """A second network and the algorithm that reads one are two sections, and neither means anything alone.

        One way round, the teacher is built, loaded from its file and carried through the run with nothing
        ever asking it anything. The other, the learner is built without the one thing it exists for —
        which does fail, but where a learner is assembled and in the words of a missing argument, rather
        than here naming the section to write.

        Both halves read the name the run wrote, so a learner reached by ``_target_`` is paired with
        nothing and accused of nothing: that import path may well be this very learner, and a refusal
        reading it would call a correct declaration wrong.
        """
        declared = self.learner.name
        if self.teacher is not None and declared is not None and declared != DISTILLATION:
            raise ValueError(
                f"`teacher` declares a second network to learn from, and `learner` is {self.learner.spelled!r}, "
                f"which learns from the data alone: the teacher would be built, read from its file and "
                f"carried through the run with nothing to ask it. Declare `learner: {{name: {DISTILLATION}}}`, "
                f"or drop the teacher."
            )
        if declared == DISTILLATION and self.teacher is None:
            raise ValueError(
                f"`learner` is {DISTILLATION!r} and there is nothing to distil from. Declare a `teacher` — the "
                f"model it is, and `checkpoint_path` for the run whose weights it answers with — or "
                f"`learner: {{name: standard}}` to learn from the data alone."
            )

    def _refuse_freezing_what_this_run_adapts(self) -> None:
        """Held still, a delta never moves, and the run trains its heads alone while the declaration says otherwise.

        Measured on a resnet adapted at its convolutions: with ``freeze`` over the same module, none of
        the thirty-four tensors of the delta are left learning, and nothing anywhere says so — the run
        trains, logs, keeps an epoch and ships it. The pair is redundant at best, because attaching a
        delta already holds the weights beneath it still; what it costs at worst is the whole run.

        Read by name, as the pairing above is: a callback reached by ``_target_`` declares no name to
        recognise, and the limitation is the one already accepted for the rate monitor.
        """
        adapted = str(self.adapter.params.get("module", "")) if self.adapter is not None else ""
        if not adapted:
            return
        for one in self.callbacks:
            if one.name != FREEZE:
                continue
            held = [str(path) for path in one.params.get("modules") or () if _reaches(str(path), adapted)]
            if held:
                raise ValueError(
                    f"`adapter` adds parameters under {adapted!r} and `callbacks` freezes "
                    f"{', '.join(repr(path) for path in held)}: the delta would be held still along with the "
                    f"weights it was added to, and nothing under {adapted!r} would learn at all. Attaching a "
                    f"delta already holds those weights still — drop the freeze, or freeze parts the adapter "
                    f"does not reach."
                )

    def _refuse_watching_a_rate_with_nothing_recording(self) -> None:
        """A callback that only reports needs somewhere to report to, and the two are separate sections.

        Left alone, Lightning refuses this itself — but at ``on_train_start``, after the sources have been
        read, the encoders fitted and the cache warmed, and in words naming ``LearningRateMonitor``, the
        ``Trainer`` and its ``logger``: three things that appear nowhere in what the run declared.
        """
        if self.tracker is None and any(one.name == LEARNING_RATE_MONITOR for one in self.callbacks):
            raise ValueError(
                f"callbacks declares {LEARNING_RATE_MONITOR!r} and tracker is none, so there is nowhere to "
                "write a learning rate. Either declare a tracker — `tracker=csv` keeps the numbers in the "
                "run's own directory — or run without the callbacks that report: `callbacks=none`. A list "
                "cannot be edited from the command line, because Hydra will not force-add to a group."
            )

    def _refuse_heads_declared_twice(self) -> None:
        """A head is a task's declaration; the model section only selects the network, or brings its own.

        Two ways to get this wrong, and they are one mistake: writing ``heads`` in the model section,
        and declaring a head for a task when the model reached by ``_target_`` arrives whole — head,
        decoding and all — so it composes none. Either way a declared head would never be built, and
        the run would report numbers for a recipe nobody ran.
        """
        if "heads" in self.model.params:
            raise ValueError("Declare heads once, under tasks; the model section only selects the network.")
        if self.model.import_path is None:
            return
        declared = sorted(name for name, task in self.tasks.items() if task.head is not None)
        if declared:
            raise ValueError(
                f"{self.model.spelled!r} is a whole model and brings its own heads, so the head declared for "
                f"{', '.join(declared)} would never be built. Drop it, or declare a backbone to compose onto."
            )

    def _refuse_inputs_that_disagree(self) -> None:
        """``data.inputs`` binds a name to a column; ``preprocessing.inputs`` gives that name an encoder.

        Two declarations, one vocabulary. Left to itself the mismatch surfaces on the first batch, inside
        a loader worker, after the sources were read and the encoders fitted.
        """
        bound = set(self.data.params.get("inputs", {}))
        encoded = set(self.preprocessing.inputs or {})
        if bound and bound != encoded:
            raise ValueError(
                f"The inputs a run binds to columns and the inputs it encodes are different names: "
                f"data.inputs has {', '.join(sorted(bound)) or 'none'}, preprocessing.inputs has "
                f"{', '.join(sorted(encoded)) or 'none'}. They name the same values."
            )
