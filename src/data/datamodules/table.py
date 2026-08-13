"""The data orchestrator: sources → split → fit encoders → profile → datasets."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, override

import pandas as pd
from torch.utils.data import ConcatDataset

from src.core.entities import DataProfile, DatasetStatistics, Distribution, Sample
from src.core.ports import DataModule, SampleTransform, require_stage
from src.core.taxonomy import Stage
from src.data.cache import BYTES_PER_GIB, CacheUsage, LoaderCache
from src.data.dataset import TableDataset
from src.data.schema import DataSchema, InputColumn, TargetColumn
from src.data.sources import Table, TableSource
from src.data.split import Splitter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceWithTransforms:
    """One source and the transforms its own rows take, wherever they land.

    Combining datasets that want different handling is what this is for: a
    clean set beside a noisy one, a synthetic set that should not be augmented
    a second time. A declared transform *replaces* the stage's for those rows
    rather than extending it — which is what lets a source be augmented less,
    not only more — so it has to end the way the stage transform does.

    Stages left undeclared fall back to the stage transform, so a source that
    only differs in training says only that.
    """

    source: TableSource
    transforms: Mapping[Stage, SampleTransform] = field(default_factory=dict)


type SourceForm = (
    TableSource
    | SourceWithTransforms
    | Sequence[TableSource | SourceWithTransforms]
    | Mapping[Stage, TableSource | SourceWithTransforms | Sequence[TableSource | SourceWithTransforms]]
)
"""Every way the rows of a run can be declared: one source, several, or one set per stage."""

type StageDataset = TableDataset | ConcatDataset[Sample]
"""What one stage hands back: its single dataset, or the sources it combines.

Named rather than widened to ``Dataset`` so ``len(module.dataset(stage))`` keeps
type-checking — the size of a stage is what callers ask for most.
"""


class TableDataModule(DataModule):
    """The table-driven ``DataModule``: annotation rows in, per-stage datasets out.

    ``setup`` is the hinge of experiment assembly: encoders are fitted on the
    train split only (no leakage from val/test), the facts they infer land in
    the ``DataProfile``, and only then can tasks and heads be built with
    concrete output sizes.

    A stage may draw on several sources. Encoders still fit on all of its train
    rows at once, so a vocabulary spans every source; only the transforms stay
    per source, which is why a stage is a concatenation of datasets rather than
    one dataset over a concatenated table.
    """

    def __init__(
        self,
        source: SourceForm,
        schema: DataSchema,
        splitter: Splitter | None = None,
        transforms: Mapping[Stage, SampleTransform] | None = None,
        cache: LoaderCache | None = None,
    ) -> None:
        self._sources = _routed(source, splitter)
        self._schema = schema
        self._transforms = dict(transforms) if transforms is not None else {}
        self._cache = cache
        self._datasets: dict[Stage, StageDataset] | None = None
        self._statistics = DatasetStatistics()

    @override
    def setup(self, profile: DataProfile) -> None:
        """Read the rows of each stage, fit encoders on train, and record the facts."""
        stages = self._read()
        if Stage.TRAIN not in stages:
            available = ", ".join(stages) or "none"
            raise ValueError(
                f"No train rows: encoders are fitted on the train stage only. Stages present: {available}."
            )
        self._fit_encoders(stages[Stage.TRAIN], profile)
        if self._cache is not None:
            self._warm_cache(self._cache, stages)
        self._datasets = {stage: self._build_stage(stage, rows) for stage, rows in stages.items()}
        self._statistics = self._describe(stages)

    def _warm_cache(self, cache: LoaderCache, stages: dict[Stage, list[_SourceRows]]) -> None:
        """Read the repeating stages once, here in the parent process.

        Train and val are read every epoch; test is read once, so memory spent
        on it buys nothing. This runs before ``DataLoader`` forks, which is what
        lets every worker share one set of decoded pixels. Any column is warmed by
        its loader — for a target that is the encoder's pre-transform half, the
        same call the dataset itself makes.

        Labels follow the cache's own namespaces (``input/image``), so what the
        bar says and what the store holds cannot drift apart. Per-column sizes
        are byte deltas around each ``warm`` — measured, not parsed out of keys.
        """
        taken: dict[str, int] = {}
        for stage in (Stage.TRAIN, Stage.VAL):
            for rows in stages.get(stage, []):
                # No scoping here: each loader carries its own scoped view of the cache,
                # applied where it was built. Warming just drives the loaders.
                for label, column in self._labelled_columns():
                    before = cache.usage().used_bytes
                    cache.warm(rows.table[column.column], column.loader, f"Caching {stage}: {label}")
                    taken[label] = taken.get(label, 0) + cache.usage().used_bytes - before
        _log_cache_summary(cache.usage(), taken)

    def _labelled_columns(self) -> list[tuple[str, InputColumn | TargetColumn]]:
        """Every column the dataset reads, labelled the way the cache scopes it."""
        return [
            *((f"input/{name}", column) for name, column in self._schema.inputs.items()),
            *((f"auxiliary_input/{name}", column) for name, column in self._schema.auxiliary_inputs.items()),
            *((f"target/{name}", column) for name, column in self._schema.targets.items()),
        ]

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
        for destination, declared in self._sources:
            table = declared.source.read()
            divided = {destination: table} if isinstance(destination, Stage) else destination(table)
            for stage, rows in divided.items():
                gathered.setdefault(stage, []).append(_SourceRows(rows, declared.transforms.get(stage)))
        return gathered

    def _fit_encoders(self, train: list[_SourceRows], profile: DataProfile) -> None:
        """Learn each target's encoding from the train rows of every source, and record it.

        Every source at once, so a vocabulary spans them all: fitted on one, a
        label from another would be rejected the first time it was encoded.
        """
        table = pd.concat([rows.table for rows in train], ignore_index=True)
        for task, target_column in self._schema.targets.items():
            target_column.encoder.fit(table[target_column.column])
            profile.record(task, target_column.encoder.facts())

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


def _routed(source: SourceForm, splitter: Splitter | None) -> list[tuple[Stage | Splitter, SourceWithTransforms]]:
    """Each declared source beside its destination — a pinned stage, or the splitter.

    The one destination shape is what lets both dataset layouts read the same
    way: rows declared under a stage carry that stage, rows that have to be
    divided carry the run's splitter. Reading stays deferred until ``setup`` —
    the rows are touched once the run is seeded, not while the experiment is
    still being assembled.
    """
    if isinstance(source, Mapping):
        if splitter is not None:
            raise ValueError(
                "Per-stage sources are already divided into stages, so a split has nothing to divide. "
                "Remove the 'data.split' section, or declare one source for it to divide."
            )
        return [(stage, declared) for stage, sources in source.items() for declared in _listed(sources)]
    if splitter is None:
        raise ValueError(
            "One source has to be divided into stages: declare a 'data.split' with per-stage "
            "fractions, or declare per-stage sources ({train: ..., val: ...}) that are already divided."
        )
    return [(splitter, declared) for declared in _listed(source)]


def _listed(
    declared: TableSource | SourceWithTransforms | Sequence[TableSource | SourceWithTransforms],
) -> list[SourceWithTransforms]:
    """Whatever was declared in one position, as the sources it stands for."""
    sources = declared if isinstance(declared, Sequence) else [declared]
    return [source if isinstance(source, SourceWithTransforms) else SourceWithTransforms(source) for source in sources]


def _log_cache_summary(usage: CacheUsage, taken: Mapping[str, int]) -> None:
    """One closing line for the whole warm-up, instead of one per column per stage.

    Only this caller knows when every stage and column is done, which is why the
    summary lives here and not in ``warm``. The breakdown answers "who took how
    much"; the second line appears only when the budget actually turned files
    away, and says what that means for the epochs to come.
    """
    breakdown = ", ".join(f"{label} {spent / BYTES_PER_GIB:.2f} GiB" for label, spent in taken.items() if spent > 0)
    log.info(
        "Cache holds %d file(s) — %.2f of %.2f GiB%s.",
        usage.files,
        usage.used_bytes / BYTES_PER_GIB,
        usage.capacity_bytes / BYTES_PER_GIB,
        f" ({breakdown})" if breakdown else "",
    )
    if usage.declined:
        log.info(
            "Cache budget full: %d file(s) did not fit and will be read from disk each epoch.",
            usage.declined,
        )
