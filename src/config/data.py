"""The data section: annotation sources, model inputs, and the stage split."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.components import ComponentConfig, TransformConfig
from src.core.taxonomy import Stage

DEFAULT_INPUT_LOADER = "image"
"""Registry key of the loader an input gets when config names none.

A default rather than a second syntax: the field keeps its one shape, and
omitting it is what a vision experiment does most of the time. Kept in step
with the ``input_loader_registry`` by a test rather than by an import — config declares
names, it does not reach into the data layer to resolve them.
"""

DEFAULT_AUXILIARY_LOADER = "mask"
"""Registry key of the loader an *auxiliary* input gets when config names none.

Different from the model-input default because the common case differs: what rides
beside the image for the augmentations to read is a mask, and reading it as a picture
would have geometry interpolate its edges and ``Normalize`` rewrite its values.
"""


InputLoaderConfig = ComponentConfig
"""The loader turning one cell into raw data: a registry name ('image') or an import path.

Not ``LoaderConfig`` — that name belongs to the DataLoader knobs in the
training section, and the port this one builds is ``InputLoader``.
"""

CacheConfig = ComponentConfig
"""The loader cache to build ('ram'), or an import path to one of your own."""


class InputColumnConfig(BaseModel):
    """One model input: which table column holds it and how to load a cell."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(
        min_length=1,
        description="Table column holding this input — an image path, a caption, a file name.",
    )
    loader: InputLoaderConfig = Field(
        default_factory=lambda: InputLoaderConfig(name=DEFAULT_INPUT_LOADER),
        description=(
            "How a cell becomes raw data: a registry name ('image') or an import path. Defaults to "
            "the image loader, this being a vision framework; declare it for anything else, or to "
            "give the image loader arguments such as a 'root' path."
        ),
    )


class AuxiliaryInputColumnConfig(InputColumnConfig):
    """One auxiliary input: a column the augmentations read and the model never sees.

    The one difference from a model input is the loader's default — the ``mask``
    loader, masks being what rides here. Declare a loader to read anything else.
    """

    loader: InputLoaderConfig = Field(
        default_factory=lambda: InputLoaderConfig(name=DEFAULT_AUXILIARY_LOADER),
        description=(
            "How a cell becomes raw data: a registry name or an import path. Defaults to "
            "the mask loader — one grayscale plane, sampled nearest-neighbour by geometry "
            "and left alone by Normalize."
        ),
    )


class SourceConfig(BaseModel):
    """One annotation source, optionally with the transforms its own rows take.

    Written as a bare path most of the time; the object form is for combining
    datasets that need different handling — a clean set beside a noisy one, a
    synthetic set that should not be augmented twice::

        source:
          - data/clean.csv
          - path: data/noisy.csv
            transforms:
              train: {_target_: src.transforms.AlbumentationsTransform, transforms: [...]}

    ``transforms`` keeps the shape of the global section — always per stage — so
    there is one thing to learn rather than a second form for this position.
    A stage left out falls back to the global transform for that stage.
    """

    model_config = ConfigDict(extra="forbid")

    path: str | list[str] = Field(description="File(s) of this source; several are concatenated.")
    format: str | None = Field(
        None,
        description=(
            "Reader for these files ('csv', 'json'); inferred from the extension when omitted. "
            "Declare it when the extension does not say — a '.txt' holding CSV — or for a format "
            "you registered whose suffix is not known. It belongs to the source rather than the "
            "run, so combined datasets may be stored differently from one another."
        ),
    )
    transforms: dict[Stage, TransformConfig] | None = Field(
        None,
        description=(
            "Per-stage transform for this source's rows, replacing the global one where declared. "
            "Replacing rather than extending is what lets a source be augmented less, not only "
            "more — so a declared pipeline must end the way the global one does, in normalisation "
            "and a tensor."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_a_bare_path(cls, value: object) -> object:
        """``source: rows.csv`` is the same declaration as ``source: {path: rows.csv}``."""
        return {"path": value} if isinstance(value, str) else value


class SplitConfig(BaseModel):
    """Per-stage row fractions; must sum to 1."""

    model_config = ConfigDict(extra="forbid")

    train: float = Field(gt=0, description="Share of rows used for training.")
    val: float = Field(gt=0, description="Share of rows used for validation.")
    test: float = Field(
        ge=0,
        description=(
            "Share of rows held out for the final test. Zero declares a run that has none: no test "
            "rows are cut, and the test stage then runs on the validation set — saying so once, "
            "because those metrics are computed on the rows the checkpoint was selected on and are "
            "optimistic for that reason. Train and val must stay positive: encoders are fitted on "
            "train, and val is what a zero test share falls back to."
        ),
    )
    seed: int = Field(
        42,
        description=(
            "Shuffle seed of the split. Deliberately separate from the experiment's 'seed': it fixes "
            "which samples land in each stage, not what happens inside a run. Keeping them apart is "
            "what makes a seed sweep meaningful — five runs at different experiment seeds must share "
            "one test set, or their metrics are not comparable. Change this only when the partition "
            "itself should change."
        ),
    )
    stratify_by: str | None = Field(
        None,
        description=(
            "Column whose distribution every stage must reproduce; None → a plain random split. "
            "Worth setting whenever the target is imbalanced: it stops a small validation set from "
            "drawing too few rows of the rare class. Repeating values are balanced as classes, a "
            "numeric column with more distinct values than 'stratify_bins' by quantile."
        ),
    )
    stratify_bins: int = Field(
        10,
        gt=1,
        description="Quantile count used when 'stratify_by' names a continuous column.",
    )
    stratify_separator: str = Field(
        ",",
        min_length=1,
        description=(
            "Separator splitting a multi-label cell ('cat,dog') into labels. Cells carrying more "
            "than one label switch stratification to the iterative algorithm, which holds each "
            "label's rate steady instead of each combination's. Match the multi-label encoder's "
            "own separator when the target column is the one being stratified."
        ),
    )
    group_by: str | None = Field(
        None,
        description=(
            "Column identifying rows that must stay in one stage — a patient, a video, a source "
            "image. Set it whenever rows are not independent: without it the same patient lands in "
            "train and in test, and the test metric measures memorisation rather than "
            "generalisation. Whole groups move together, so stage sizes approximate the fractions."
        ),
    )

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> SplitConfig:
        """Fractions that do not cover the table silently leave rows in no stage at all."""
        total = self.train + self.val + self.test
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Split fractions must sum to 1, got {total}.")
        return self

    @model_validator(mode="after")
    def _one_way_of_dividing(self) -> SplitConfig:
        """The two guards pull against each other, so honouring both is a design, not a default."""
        if self.stratify_by is not None and self.group_by is not None:
            raise ValueError(
                "'stratify_by' and 'group_by' cannot be combined yet: balancing classes needs to "
                "move single rows, keeping groups intact forbids it, and reconciling the two is a "
                "deliberate design rather than a default. Pick the risk that matters more here — "
                "leakage between related rows, or an unbalanced stage."
            )
        return self

    def fractions(self) -> dict[Stage, float]:
        """The split as the ``Stage``-keyed mapping a splitter consumes.

        A stage with a zero share is left out rather than handed over as a zero.
        Splitters give their *last* stage whatever the flooring left over, and test
        is last — so a zero share passed along would cut one or two rows instead of
        none, and a run that declared no test set would report a test metric
        computed on two samples. Left out, the stage simply does not exist, which is
        the answer ``TrainingData`` already knows how to act on.
        """
        declared = {Stage.TRAIN: self.train, Stage.VAL: self.val, Stage.TEST: self.test}
        return {stage: share for stage, share in declared.items() if share > 0}


class DataConfig(BaseModel):
    """Where the annotation rows come from and how their columns feed the model.

    Task targets are deliberately absent: a target column and its encoder are declared
    once, on the task, and the data schema is derived from the tasks at assembly.

    A section with no ``inputs`` is a valid declaration, not a mistake: a vendor pipeline
    reads its images from its own descriptor and has no columns at all. That a *table*
    needs at least one input column is true of the table, so it is stated where the
    table's schema is built, beside the other rules only a table has.

    The form of ``source`` picks between the two ways a dataset arrives::

        source: data/annotations.csv               # one table, 'split' divides it
        split: {train: 0.7, val: 0.15, test: 0.15}

        source:                                    # already divided upstream
          train: data/train.csv
          val: data/val.csv
          test: data/test.csv

    The second form matters because a partition is often not ours to make: a
    competition ships one, a temporal or per-patient split is decided before the
    data reaches us, and re-dividing those rows by fractions would quietly undo
    the very separation they encode.
    """

    model_config = ConfigDict(extra="forbid")

    source: SourceConfig | list[SourceConfig] | dict[Stage, SourceConfig | list[SourceConfig]] = Field(
        description=(
            "Where the annotation rows come from. A path → one dataset that 'split' divides into "
            "stages. A list → several datasets combined, each divided by the same fractions so "
            "every one is represented in every stage, and each free to carry its own transforms. "
            "A stage-keyed mapping → the rows are already divided and 'split' must be absent."
        ),
    )
    inputs: dict[str, InputColumnConfig] = Field(
        description="Model inputs by name; the name is how a batch and a backbone refer to them.",
    )
    auxiliary_inputs: dict[str, AuxiliaryInputColumnConfig] = Field(
        default_factory=dict,
        description=(
            "Columns loaded for the augmentations alone — a mask that bounds a colour shift. "
            "Loaded like inputs and carried through the sample transforms with the geometry "
            "their loader declares (the default is the mask loader: one grayscale plane, "
            "nearest-neighbour geometry, untouched by Normalize), but never collated: the "
            "model does not see them and no batch memory is spent on them. A mask the model "
            "should consume is a regular input with loader {name: mask}."
        ),
    )
    split: SplitConfig | None = Field(
        None,
        description="How to divide one source into stages. Required for a single source, forbidden for per-stage ones.",
    )
    cache: CacheConfig | None = Field(
        None,
        description=(
            "Holds decoded files in RAM so an epoch does not decode what the last one did, e.g. "
            "{name: ram, max_gib: 8}. Absent → no cache. Warmed once before the data loader forks "
            "its workers, so the pixels are shared between them rather than decoded in each."
        ),
    )
    max_samples: int | float | None = Field(
        None,
        gt=0,
        description=(
            "Cap on the rows read — a count, or a share in (0, 1]. For quick iteration on a small "
            "slice of a large dataset. Rows are drawn at random rather than off the top, since "
            "annotation files routinely arrive grouped by class or ordered by date. Applies before "
            "the split for a single source, and per stage for per-stage ones."
        ),
    )

    @model_validator(mode="after")
    def _pair_the_source_form_with_the_split(self) -> DataConfig:
        """Per-stage sources are used as given, so they must name a train stage and transform only their own."""
        if not isinstance(self.source, dict):
            # Whether one source can stand without a `split` is the table's question, and
            # `TableDataModule` already answers it. Asked here as well it would also refuse
            # a vendor pipeline, whose descriptor names its own stages.
            return self
        if Stage.TRAIN not in self.source:
            declared = ", ".join(sorted(self.source)) or "none"
            raise ValueError(f"Per-stage sources need a '{Stage.TRAIN}' entry; got: {declared}.")
        for stage, declared_sources in self.source.items():
            listed = declared_sources if isinstance(declared_sources, list) else [declared_sources]
            for source in listed:
                foreign = sorted(str(name) for name in (source.transforms or {}) if name != stage)
                if foreign:
                    raise ValueError(
                        f"Source '{source.path}' is declared under stage '{stage}' but carries "
                        f"transforms for {', '.join(foreign)}; its rows never reach those stages."
                    )
        return self
