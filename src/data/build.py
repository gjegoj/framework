"""Building the table pipeline from its declaration: source, schema, split and transforms.

The data capability reads its own section of the config here, and the target half of the
tasks section — which column a task reads and how it is encoded — because the schema is
derived from the tasks that own their targets. Nothing else of the experiment is read.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.config import ComponentConfig
from src.config.instantiate import instantiate, resolve_params, resolve_target

# At runtime, not under TYPE_CHECKING: whether a built transform takes geometry is an isinstance.
from src.core.ports import GeometryAware
from src.core.taxonomy import Geometry, Stage
from src.data.cache import cached
from src.data.datamodules import DeclaredSource, TableDataModule
from src.data.encoders import FileTargetEncoder, VocabularyTargetEncoder
from src.data.registry import cache_registry, input_loader_registry, table_source_registry, target_encoder_registry
from src.data.schema import ColumnRole, DataSchema, InputColumn, TargetColumn
from src.data.sources import format_of
from src.data.split import group_split, random_split, stratified_split

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from src.config.data import CacheConfig, DataConfig, InputColumnConfig, SourceConfig, SplitConfig
    from src.config.tasks import TaskConfig
    from src.core.ports import SampleTransform
    from src.data.cache import LoaderCache
    from src.data.encoders import TargetEncoder
    from src.data.loaders import InputLoader
    from src.data.sources import TableSource
    from src.data.split import Splitter
    from src.tasks import TaskKind

log = logging.getLogger(__name__)


def build_pipeline(
    data: DataConfig,
    tasks: Mapping[str, TaskConfig],
    kinds: Mapping[str, TaskKind],
    transforms: Mapping[Stage, ComponentConfig] | None,
) -> TableDataModule:
    """Source, schema, split and transforms, wired into one table pipeline.

    ``kinds`` are the tasks' kinds, built once by the composition root: each says which
    encoding its loss needs, and the schema is derived from that and the declarations.
    """
    cache = build_cache(data.cache)
    schema = build_schema(data, tasks, kinds, cache)
    return TableDataModule(
        sources=_sources(data, schema),
        schema=schema,
        splitter=build_splitter(data.split) if data.split is not None else None,
        transforms=build_transforms(transforms, schema),
        cache=cache,
        max_samples=data.max_samples,
    )


def build_cache(declared: CacheConfig | None) -> LoaderCache | None:
    """The declared cache, or ``None`` when the section is absent."""
    if declared is None:
        return None
    built: LoaderCache = instantiate(declared, cache_registry)
    return built


def build_schema(
    data: DataConfig, tasks: Mapping[str, TaskConfig], kinds: Mapping[str, TaskKind], cache: LoaderCache | None = None
) -> DataSchema:
    """Map table columns to model inputs and per-task targets.

    Targets come from the tasks that own them: a target column and its encoder are declared
    once, in the task, and the schema is derived from that. A cache, when there is one, is
    applied here: input loaders are wrapped, while an encoder that reads files behind a
    loader of its own is handed the cache (``FileTargetEncoder.use_cache``) and wraps that
    loader itself. Namespaces are scoped per column and qualified by role — the ``(role, name)``
    pair — because an input and a task may legally share a name; a test pins the pairs to the
    schema's own columns.
    """
    return DataSchema(
        inputs=_input_columns(data.inputs, cache, ColumnRole.INPUT),
        targets={
            name: TargetColumn(
                column=str(declared.target),
                encoder=build_target_encoder(name, declared, kinds[name], _scoped(cache, ColumnRole.TARGET, name)),
            )
            for name, declared in tasks.items()
            if declared.target is not None
        },
        auxiliary_inputs=_input_columns(data.auxiliary_inputs, cache, ColumnRole.AUXILIARY_INPUT),
    )


def build_target_encoder(
    name: str, declared: TaskConfig, kind: TaskKind, cache: LoaderCache | None = None
) -> TargetEncoder:
    """The declared encoder, or the kind's default, handed what its base says it takes.

    A kind knows which encoding its loss needs, so declaring an encoder is an override, not
    a duty. What an encoder takes beyond its declaration it says by its base: a
    ``VocabularyTargetEncoder`` is handed the task's ``classes`` and refused without them, any other
    encoder refuses a declared vocabulary, a ``FileTargetEncoder`` reads through the cache.
    The vocabulary has one spelling — on the task — so one written inside the encoder's
    declaration is refused by name, as a head's sizes are. Nothing is matched by parameter
    name, so nothing is dropped in silence.
    """
    component = (
        declared.target_encoder if declared.target_encoder is not None else _default_encoder(name, declared, kind)
    )
    _refuse_a_vocabulary_inside(name, component)
    factory = resolve_target(component, target_encoder_registry)
    params = resolve_params(component)
    reads_a_vocabulary = isinstance(factory, type) and issubclass(factory, VocabularyTargetEncoder)
    if reads_a_vocabulary:
        try:
            encoder: TargetEncoder = factory(classes=_required_classes(name, declared), **params)
        except ValueError as error:
            raise ValueError(f"Task '{name}': {error}") from None
    else:
        _refuse_classes_nobody_reads(name, declared, component)
        encoder = factory(**params)
    if cache is not None and isinstance(encoder, FileTargetEncoder):
        encoder.use_cache(cache)
    return encoder


def _default_encoder(name: str, declared: TaskConfig, kind: TaskKind) -> ComponentConfig:
    """The kind's own encoder as a declaration — or the question a targetless kind asks back."""
    if kind.default_encoder is None:
        raise ValueError(
            f"Task '{name}' declares target '{declared.target}', but kind '{declared.kind.spelled}' takes its "
            f"supervision from the structure of a batch rather than from a column. Drop the target, or "
            f"declare the 'target_encoder' that makes sense of it."
        )
    return ComponentConfig(name=kind.default_encoder)


def _refuse_a_vocabulary_inside(name: str, component: ComponentConfig) -> None:
    if "classes" in component.params:
        raise ValueError(
            f"'{component.spelled}' declares classes, which task '{name}' declares; the vocabulary has one "
            f"spelling, on the task. Drop it from the encoder."
        )


def _required_classes(name: str, declared: TaskConfig) -> dict[int, str]:
    """The task's vocabulary, which an encoder that reads one cannot do without.

    The index space of a model's outputs is a declaration, not a fact of whichever rows a
    split or a sample cap left in train: learned, it shrank when a rare class dropped out
    and the checkpoint keyed on it stopped fitting in silence.
    """
    if declared.classes is None:
        raise ValueError(
            f"Task '{name}' needs 'classes': its target encoder reads a vocabulary, and the index space must "
            f"be declared rather than learned from the rows. Declare it on the task, "
            f"e.g. classes: {{0: background, 1: defect}}."
        )
    return declared.classes


def _refuse_classes_nobody_reads(name: str, declared: TaskConfig, component: ComponentConfig) -> None:
    """A user's declaration may not be dropped in silence; only a derived fact may."""
    if declared.classes is not None:
        raise ValueError(
            f"Task '{name}' declares classes, but its target encoder '{component.spelled}' carries no "
            f"vocabulary. Declare an encoder that does (label, multilabel, mask, boxes), or drop 'classes'."
        )


def _input_columns(
    declared: Mapping[str, InputColumnConfig], cache: LoaderCache | None, role: ColumnRole
) -> dict[str, InputColumn]:
    """One declared section of input columns, built and cache-scoped under its role.

    Serves ``inputs`` and ``auxiliary_inputs`` alike: the two differ only in where their
    values go after loading, which is the schema's business, not this builder's.
    """
    return {name: _input_column(name, column, _scoped(cache, role, name)) for name, column in declared.items()}


def _input_column(name: str, column: InputColumnConfig, cache: LoaderCache | None) -> InputColumn:
    """One built column: the declared loader, wrapped for the cache, and what it reads.

    ``geometry`` is taken from the loader *before* the wrapping — ``cached`` returns a bare
    closure, so asking the wrapped loader would silently answer with the default and a
    cached mask input would quietly get picture treatment in the pipeline.
    """
    loader: InputLoader = instantiate(column.loader, input_loader_registry)
    return InputColumn(
        column=column.column,
        loader=cached(loader, cache) if cache is not None else loader,
        geometry=_geometry_of(name, loader),
    )


def _geometry_of(name: str, loader: InputLoader) -> Geometry:
    """What the loader declares, or a picture — said out loud, because it is a substitution.

    The port admits any callable, and a plain function has no class to carry a geometry; the
    built-in loaders declare theirs as a ``ClassVar``. A column left to the default is resized
    and normalised like a photograph, which is wrong for a vector or a map, so the run names
    the column and the loader rather than letting the default win silently.
    """
    declared: Geometry | None = getattr(loader, "geometry", None)
    if declared is not None:
        return declared
    log.info(
        "Input '%s': loader %s declares no geometry; treated as a picture (resized, normalised). "
        "Declare `geometry` on the loader for anything else.",
        name,
        type(loader).__name__,
    )
    return Geometry.IMAGE


def _scoped(cache: LoaderCache | None, *namespace: str) -> LoaderCache | None:
    """The cache this column may write into, or ``None`` when there is no cache."""
    return cache.scoped(*namespace) if cache is not None else None


def build_transforms(
    declared: Mapping[Stage, ComponentConfig] | None, schema: DataSchema
) -> dict[Stage, SampleTransform]:
    """The experiment's per-stage transforms, with evaluation declared once.

    The two eval stages complete each other — a ``test`` pipeline left undeclared is the
    ``val`` one, said out loud — and a section declaring only ``train`` is refused by name:
    evaluating on pictures that were never resized or normalised is not a measurement,
    and measured, a missing stage used to mean exactly that, in silence. A source's own
    transforms override only the stages they name and go through ``build_stage_transforms``.
    """
    built = build_stage_transforms(declared, schema)
    if not built:
        return built
    evaluation = [stage for stage in (Stage.VAL, Stage.TEST) if stage in built]
    if not evaluation:
        raise ValueError(
            "The transforms section declares a train pipeline and no evaluation pipeline: val and test "
            "would run on pictures never resized or normalised. Declare 'val' (test takes it when absent)."
        )
    for stage in (Stage.VAL, Stage.TEST):
        if stage not in built:
            built[stage] = built[evaluation[0]]
            log.info("No '%s' transforms declared: the '%s' pipeline serves %s too.", stage, evaluation[0], stage)
    return built


def build_stage_transforms(
    declared: Mapping[Stage, ComponentConfig] | None, schema: DataSchema
) -> dict[Stage, SampleTransform]:
    """Per-stage transforms as declared, each told which arrays follow the image.

    None of ``inputs``, ``targets`` or ``auxiliary_inputs`` is written by hand: each comes
    from the ``geometry`` its own loader or encoder declares, and all three are bound in one
    call (``GeometryAware.with_geometry``), so a mask or a boxes column cannot fall out of
    step with its image. A ``NONE``-geometry value is left out rather than bound, on every
    side: a label column must not enter the pipeline uninvited, and an embedding input must
    not be normalised as if it were light. A transform without the port is built as
    declared and needs to know nothing about any of this.
    """
    if not declared:
        return {}
    geometry: dict[str, Any] = {
        "inputs": _pixel_geometries({name: column.geometry for name, column in schema.inputs.items()}),
        "targets": _pixel_geometries({name: column.encoder.geometry for name, column in schema.targets.items()}),
        "auxiliary_inputs": _pixel_geometries(
            {name: column.geometry for name, column in schema.auxiliary_inputs.items()}
        ),
    }
    built: dict[Stage, SampleTransform] = {}
    for stage, component in declared.items():
        transform = instantiate(component)
        built[stage] = transform.with_geometry(**geometry) if isinstance(transform, GeometryAware) else transform
    return built


def _pixel_geometries(declared: Mapping[str, Geometry]) -> dict[str, Geometry]:
    return {name: geometry for name, geometry in declared.items() if geometry is not Geometry.NONE}


def build_splitter(split: SplitConfig) -> Splitter:
    """The declared way of dividing rows into stages; plain random unless a column is named."""
    if split.group_by is not None:
        return group_split(split.fractions(), by=split.group_by, seed=split.seed)
    if split.stratify_by is not None:
        return stratified_split(
            split.fractions(),
            by=split.stratify_by,
            seed=split.seed,
            bins=split.stratify_bins,
            separator=split.stratify_separator,
        )
    return random_split(split.fractions(), seed=split.seed)


def _sources(data: DataConfig, schema: DataSchema) -> list[DeclaredSource]:
    """Every declared source with the stage it is pinned to — none where the splitter divides it."""
    if isinstance(data.source, dict):
        return [
            _source_for(source, schema, stage)
            for stage, declared in data.source.items()
            for source in _listed(declared)
        ]
    return [_source_for(source, schema, None) for source in _listed(data.source)]


def _listed(declared: SourceConfig | list[SourceConfig]) -> list[SourceConfig]:
    """What one position declares: a single source, or several to combine."""
    return declared if isinstance(declared, list) else [declared]


def _source_for(declared: SourceConfig, schema: DataSchema, stage: Stage | None) -> DeclaredSource:
    """One source: its format inferred unless declared, with any transforms of its own."""
    paths = declared.path if isinstance(declared.path, list) else [declared.path]
    source: TableSource = table_source_registry.create(declared.format or format_of(paths[0]), paths=paths)
    return DeclaredSource(source=source, stage=stage, transforms=build_stage_transforms(declared.transforms, schema))


def input_geometries(declared: Mapping[str, InputColumnConfig]) -> dict[str, Geometry]:
    """What each declared input is, read off its loader's class without building it.

    For the export example: a picture is shaped from ``image_size`` and ``mean``; anything else has
    no shape config can name. A loader with no class-level ``geometry`` is a picture, the same
    default ``_geometry_of`` announces when the pipeline is built.
    """
    return {
        name: getattr(resolve_target(column.loader, input_loader_registry), "geometry", Geometry.IMAGE)
        for name, column in declared.items()
    }
