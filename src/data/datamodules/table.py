"""The data orchestrator: sources → split → fit encoders → facts → datasets."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, override

import pandas as pd
from torch.utils.data import ConcatDataset

from src.core.entities import DatasetFacts, Sample, TaskFacts
from src.core.ports import SampleTransform
from src.core.taxonomy import Stage
from src.data.cache import LoaderCache
from src.data.datamodules.base import DataModule, require_stage
from src.data.dataset import TableDataset
from src.data.schema import DataSchema
from src.data.sources import Table, TableSource, capped, refuse_a_bad_cap
from src.data.split import Splitter
from src.data.statistics import DatasetStatistics, Distribution

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeclaredSource:
    """One source of rows: what it reads, where its rows land, and the transforms they take.

    ``stage`` pins the rows to one stage — a partition decided upstream is used as given;
    ``None`` leaves them to the run's splitter, and the two layouts may sit side by side. A
    declared transform *replaces* the stage's for those rows — for combining datasets that
    want different handling: a clean set beside a noisy one, a synthetic set that should not
    be augmented twice — so it has to end the way the stage transform does; undeclared
    stages fall back to the stage transform. A table handed in directly is read as it
    stands: notebooks, tests, generated data, which no config can express.
    """

    source: TableSource | Table
    stage: Stage | None = None
    transforms: Mapping[Stage, SampleTransform] = field(default_factory=dict)


type StageDataset = TableDataset | ConcatDataset[Sample]
"""What one stage hands back: its single dataset, or the sources it combines.

Named rather than widened to ``Dataset`` so ``len(module.dataset(stage))`` keeps
type-checking — the size of a stage is what callers ask for most.
"""


class TableDataModule(DataModule):
    """The table-driven ``DataModule``: annotation rows in, per-stage datasets out.

    ``setup`` is the hinge of the build: encoders fit on the train split only and validate the
    others, their facts are returned by ``setup``, and only then can heads be built with
    concrete sizes. A
    stage may draw on several sources; encoders still fit on all of its train rows at once,
    only the transforms stay per source.
    """

    # PYI041 reads 'int | float' as a redundant union; here it is the contract itself: a count, or a share.
    def __init__(
        self,
        sources: Sequence[DeclaredSource],
        schema: DataSchema,
        splitter: Splitter | None = None,
        transforms: Mapping[Stage, SampleTransform] | None = None,
        cache: LoaderCache | None = None,
        max_samples: int | float | None = None,  # noqa: PYI041
    ) -> None:
        _refuse_a_layout_the_splitter_cannot_serve(sources, splitter)
        refuse_a_bad_cap(max_samples)
        self._sources = list(sources)
        self._splitter = splitter
        self._max_samples = max_samples
        self._schema = schema
        self._transforms = dict(transforms) if transforms is not None else {}
        self._cache = cache
        self._datasets: dict[Stage, StageDataset] | None = None
        self._statistics = DatasetStatistics()

    @override
    def setup(self) -> DatasetFacts:
        """Read the rows of each stage, fit the encoders on train, and return what they learned."""
        stages = self._read()
        if Stage.TRAIN not in stages:
            available = ", ".join(stages) or "none"
            raise ValueError(
                f"No train rows: encoders are fitted on the train stage only. Stages present: {available}."
            )
        facts = self._fit_encoders(stages[Stage.TRAIN])
        self._validate_encoders(stages)
        if self._cache is not None:
            self._warm_cache(self._cache, stages)
        self._datasets = {stage: self._build_stage(stage, rows) for stage, rows in stages.items()}
        self._statistics = self._describe(stages)
        return facts

    def _warm_cache(self, cache: LoaderCache, stages: dict[Stage, list[_SourceRows]]) -> None:
        """Read the repeating stages once, here in the parent process.

        Train and val are read every epoch; test is read once, so memory spent on it buys
        nothing. Runs before ``DataLoader`` forks, so every worker shares one set of decoded
        pixels. A target column is warmed through its encoder's pre-transform half.
        """
        for stage in (Stage.TRAIN, Stage.VAL):
            for rows in stages.get(stage, []):
                # No scoping here: each loader carries its own scoped view of the cache,
                # applied where it was built. Warming just drives the loaders.
                for role, name, column in self._schema.columns_by_role():
                    cache.warm(rows.table[column.column], column.loader, f"{stage}: {role.label(name)}")
        cache.summarize()

    @override
    def dataset(self, stage: Stage) -> StageDataset:
        """Return the dataset for ``stage``; ``setup`` must have run first."""
        return require_stage(self._datasets, stage, type(self).__name__)

    @override
    def statistics(self) -> DatasetStatistics:
        """How many rows each stage holds, and what each target column looks like.

        Taken at ``setup`` from the annotation tables, before a single batch is
        loaded: the encoders are fitted by then, so each one can describe its own
        column against the vocabulary it just learned — including the classes the
        split never produced, which is the row worth reading.
        """
        return self._statistics

    def _describe(self, stages: dict[Stage, list[_SourceRows]]) -> DatasetStatistics:
        """Count the rows and ask every encoder to describe its column, per stage.

        Sources are concatenated per stage here, unlike the split: a distribution
        is about the stage a model will see, and the model sees the sources joined.
        """
        tables = {
            stage: pd.concat([rows.table for rows in listed], ignore_index=True) for stage, listed in stages.items()
        }
        targets: dict[str, dict[Stage, Distribution]] = {}
        for task, target_column in self._schema.targets.items():
            described = {
                stage: distribution
                for stage, table in tables.items()
                if (distribution := target_column.encoder.distribution(table[target_column.column])) is not None
            }
            # Kept even when empty: the report names a task that describes nothing,
            # rather than leaving the reader to notice one of their targets is gone.
            targets[task] = described
        return DatasetStatistics(rows={stage: len(table) for stage, table in tables.items()}, targets=targets)

    def _read(self) -> dict[Stage, list[_SourceRows]]:
        """Every source's rows, gathered under the stages each one reaches.

        Sources are divided *one by one* rather than as one concatenated table,
        so each is represented in every stage in its declared proportion — a
        small source cannot land wholly in train by an unlucky draw.
        """
        gathered: dict[Stage, list[_SourceRows]] = {}
        for declared in self._sources:
            # Capped per source, for the same reason: combining datasets must not shrink each to a share.
            table = capped(_rows_of(declared.source), self._max_samples)
            pinned = declared.stage
            divided = {pinned: table} if pinned is not None else self._divided(table)
            for stage, rows in divided.items():
                gathered.setdefault(stage, []).append(_SourceRows(rows, declared.transforms.get(stage)))
        return gathered

    def _divided(self, table: Table) -> Mapping[Stage, Table]:
        """The splitter's stages for a source pinned to none — construction guaranteed the splitter."""
        assert self._splitter is not None, "a source without a stage passed construction without a splitter"
        return self._splitter(table)

    def _fit_encoders(self, train: list[_SourceRows]) -> DatasetFacts:
        """Learn each target's encoding from the train rows of every source; what each learned, by task.

        Every source at once, so a vocabulary spans them all: fitted on one, a
        label from another would be rejected the first time it was encoded.
        """
        table = pd.concat([rows.table for rows in train], ignore_index=True)
        facts: dict[str, TaskFacts] = {}
        for task, target_column in self._schema.targets.items():
            facts[task] = target_column.encoder.fit(table[target_column.column]).facts()
        return facts

    def _validate_encoders(self, stages: dict[Stage, list[_SourceRows]]) -> None:
        """Every split ``fit`` never saw, checked against what it learned — refused by stage and task.

        A class misspelt in ``val.jsonl`` used to surface from ``encode`` in the first validation
        epoch; here it dies at setup, before a batch is read.
        """
        for stage, sources in stages.items():
            if stage is Stage.TRAIN:
                continue
            table = pd.concat([rows.table for rows in sources], ignore_index=True)
            for task, target_column in self._schema.targets.items():
                try:
                    target_column.encoder.validate(table[target_column.column])
                except (LookupError, ValueError, TypeError) as error:
                    raise type(error)(f"Stage '{stage}', task '{task}': {error}") from error

    def _build_stage(self, stage: Stage, sources: list[_SourceRows]) -> StageDataset:
        """One dataset per source, combined — a single source stays a single dataset."""
        if own := sum(1 for rows in sources if rows.transform is not None):
            log.info(
                "Stage '%s' draws on %d source(s), %d with their own transform. A source transform "
                "replaces the stage's for those rows, so it has to end the same way.",
                stage,
                len(sources),
                own,
            )
        datasets = [
            TableDataset(rows.table, self._schema, transform=rows.transform or self._transforms.get(stage))
            for rows in sources
        ]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


class _SourceRows(NamedTuple):
    """The rows one source contributes to one stage, and the transform they take."""

    table: Table
    transform: SampleTransform | None


def _refuse_a_layout_the_splitter_cannot_serve(sources: Sequence[DeclaredSource], splitter: Splitter | None) -> None:
    """Rows reach stages one way or the other — said at construction, not by a setup that finds no way."""
    undivided = [declared for declared in sources if declared.stage is None]
    if undivided and splitter is None:
        raise ValueError(
            "One source has to be divided into stages: declare a 'data.split' with per-stage "
            "fractions, or declare per-stage sources ({train: ..., val: ...}) that are already divided."
        )
    if splitter is not None and not undivided:
        raise ValueError(
            "Per-stage sources are already divided into stages, so a split has nothing to divide. "
            "Remove the 'data.split' section, or declare one source for it to divide."
        )


def _rows_of(source: TableSource | Table) -> Table:
    """A table as it stands, or whatever the source reads — at setup, once the run is seeded."""
    return source if isinstance(source, pd.DataFrame) else source.read()
