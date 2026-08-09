"""Building the data side of an experiment from config."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.assembly.instantiate import instantiate, resolve_target
from src.assembly.vendor import is_vendor_family
from src.core.registry import named_by
from src.data import (
    DataSchema,
    InputColumn,
    LimitedSource,
    LoaderCache,
    SourceWithTransforms,
    TableDataModule,
    TargetColumn,
    YoloDataModule,
    cached,
    group_split,
    random_split,
    stratified_split,
)
from src.data.registry import (
    cache_registry,
    input_loader_registry,
    table_source_registry,
    target_encoder_registry,
)
from src.models.registry import model_registry
from src.tasks.registry import objective_registry, topology_registry

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from src.config import ComponentConfig, ExperimentConfig
    from src.config.data import DataConfig, InputColumnConfig, SourceConfig, SplitConfig
    from src.config.tasks import TaskConfig
    from src.core.ports import DataModule, SampleTransform
    from src.core.taxonomy import Stage
    from src.data import InputLoader, Splitter, TableSource, TargetEncoder

from src.data.datamodules import SourceForm

SUFFIX_FORMATS = {".csv": "csv", ".json": "json"}
"""Table formats inferable from a file extension, mapped to ``table_source_registry`` keys."""


def build_cache(config: ExperimentConfig) -> LoaderCache | None:
    """The declared cache, or ``None`` when the section is absent."""
    if config.data.cache is None:
        return None
    built: LoaderCache = instantiate(config.data.cache, cache_registry)
    return built


def build_data_schema(config: ExperimentConfig, cache: LoaderCache | None = None) -> DataSchema:
    """Map table columns to model inputs and per-task targets.

    Targets come from the tasks that own them: a target column and its encoder
    are declared once, in the task, and the schema is derived from that — the
    single source of truth this config contract was shaped around.

    A cache, when there is one, is applied here: input loaders are wrapped,
    while an encoder that reads files behind a loader of its own is offered the
    cache as a derived value and wraps that loader itself.
    """
    # Scoped per column — one cache so every column shares one budget, scoped keys so
    # an image and a mask stored under one filename cannot serve each other's arrays.
    # Qualified by kind, because an input and a task may legally share a name.
    return DataSchema(
        inputs={
            name: InputColumn(column=column.column, loader=_build_loader(column, _for(cache, f"input/{name}")))
            for name, column in config.data.inputs.items()
        },
        targets={
            name: TargetColumn(
                column=str(task.target),
                encoder=_build_target_encoder(name, task, {"cache": _for(cache, f"target/{name}")} if cache else {}),
            )
            for name, task in config.tasks.items()
            if task.target is not None
        },
    )


def _for(cache: LoaderCache | None, name: str) -> LoaderCache | None:
    """The cache this column may write into, or ``None`` when there is no cache."""
    return cache.scoped(name) if cache is not None else None


def _build_loader(column: InputColumnConfig, cache: LoaderCache | None) -> InputLoader:
    """The declared loader, reading through the cache when there is one."""
    loader: InputLoader = instantiate(column.loader, input_loader_registry)
    return cached(loader, cache) if cache is not None else loader


def _build_target_encoder(name: str, task: TaskConfig, derived: Mapping[str, Any]) -> TargetEncoder:
    """The declared encoder, or the one this task's objective implies.

    A preset names a familiar kind of task, and the encoding its loss needs
    follows from that — so declaring an encoder is an override, not a duty.
    The two cases config still has to answer for both fail loudly here.
    """
    offered = {**derived, **({"classes": task.classes} if task.classes is not None else {})}
    if task.target_encoder is not None:
        if task.classes is not None and "classes" in task.target_encoder.params:
            # Both are user declarations; the derived channel would resolve them silently.
            raise ValueError(
                f"Task '{name}' declares classes both on the task and inside its target_encoder; "
                "declare the vocabulary once — on the task."
            )
        declared: TargetEncoder = instantiate(task.target_encoder, target_encoder_registry, **offered)
        return _honouring_declared_classes(name, task, declared)
    if topology_registry.create(task.topology).spatial_targets:
        raise ValueError(
            f"Task '{name}' needs an explicit 'target_encoder': a {task.topology} target is an "
            f"image of its own, and reading it needs the class count — for example "
            f"target_encoder: {{name: mask, num_classes: 3}}."
        )
    default = objective_registry.create(task.objective).default_target_encoder()
    if default is None:
        raise ValueError(
            f"Task '{name}' declares target '{task.target}', but objective '{task.objective}' takes "
            f"its supervision from the structure of a batch rather than from a column. Drop the "
            f"target, or declare the 'target_encoder' that makes sense of it."
        )
    factory = target_encoder_registry.get(default)
    return _honouring_declared_classes(name, task, factory(**named_by(factory, offered)))


def _honouring_declared_classes(name: str, task: TaskConfig, built: TargetEncoder) -> TargetEncoder:
    """Derived facts may be dropped silently — a user's declaration may not.

    ``classes`` travels through the same offering as other derived values, so an
    encoder that never names it would swallow the declaration whole; an encoder
    that consumed it carries the names already, before any fit.
    """
    if task.classes is not None and built.class_names is None:
        raise ValueError(
            f"Task '{name}' declares classes, but its target encoder does not carry a vocabulary. "
            "Declare an encoder that does (label, multilabel, mask), or drop 'classes'."
        )
    return built


def build_transforms(config: ExperimentConfig, schema: DataSchema) -> dict[Stage, SampleTransform]:
    """Build per-stage transforms, telling each which targets follow the image.

    ``spatial_targets`` is never written by hand: it is derived from the
    encoders (``TargetEncoder.spatial``) and passed as a derived value, so a
    mask cannot silently fall out of step with its image. A schema without
    spatial targets passes nothing, so transforms for classification runs need
    to know nothing about this.
    """
    if config.transforms is None:
        return {}
    return _build_stage_transforms(config.transforms, schema)


def _build_stage_transforms(
    declared: Mapping[Stage, ComponentConfig], schema: DataSchema
) -> dict[Stage, SampleTransform]:
    """Per-stage transforms from their declarations, wherever the declarations came from."""
    spatial = [name for name, column in schema.targets.items() if column.encoder.spatial]
    derived = {"spatial_targets": spatial} if spatial else {}
    return {stage: instantiate(component, **derived) for stage, component in declared.items()}


def build_data_module(config: ExperimentConfig) -> DataModule:
    """Source, schema, split and transforms, wired into one data module.

    Returns the port rather than the table implementation: a vendor family arrives with
    its own pipeline — YOLO reads a native ``data.yaml`` — and is recognised here the same
    way its model is, from the name in ``config.model``. One reading of one key decides
    both, so the two cannot disagree about what kind of run this is.
    """
    if is_vendor_family(config):
        return _vendor_data_module(config)
    cache = build_cache(config)
    schema = build_data_schema(config, cache)
    split = config.data.split
    return TableDataModule(
        source=_build_source(config, schema),
        schema=schema,
        splitter=build_splitter(split) if split is not None else None,
        transforms=build_transforms(config, schema),
        cache=cache,
    )


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


def _build_source(config: ExperimentConfig, schema: DataSchema) -> SourceForm:
    """The sources a run reads, in whichever of the three forms config declares."""
    declared = config.data.source
    if isinstance(declared, dict):
        return {stage: _sources_for(sources, config.data, schema) for stage, sources in declared.items()}
    return _sources_for(declared, config.data, schema)


def _sources_for(
    declared: SourceConfig | list[SourceConfig], data: DataConfig, schema: DataSchema
) -> list[SourceWithTransforms]:
    """The sources declared in one position — a single one, or several to combine."""
    listed = declared if isinstance(declared, list) else [declared]
    return [_source_for(source, data, schema) for source in listed]


def _source_for(declared: SourceConfig, data: DataConfig, schema: DataSchema) -> SourceWithTransforms:
    """One source: capped, its format inferred unless declared, with any transforms of its own."""
    paths = declared.path if isinstance(declared.path, list) else [declared.path]
    table_format = declared.format or _infer_format(paths[0])
    source: TableSource = table_source_registry.create(table_format, paths=paths)
    if data.max_samples is not None:
        # Per source, so combining datasets does not shrink each one to a share of the cap.
        source = LimitedSource(source, data.max_samples)
    transforms = _build_stage_transforms(declared.transforms, schema) if declared.transforms else {}
    return SourceWithTransforms(source=source, transforms=transforms)


def _infer_format(path: str) -> str:
    suffix = Path(path).suffix.lower()
    try:
        return SUFFIX_FORMATS[suffix]
    except KeyError:
        known = ", ".join(sorted(str(key) for key in table_source_registry))
        raise LookupError(
            f"Cannot infer the table format of '{path}'. "
            f"Give the source a 'format' explicitly; registered formats: {known}."
        ) from None


def _vendor_data_module(config: ExperimentConfig) -> DataModule:
    """The family's own pipeline, told the run's facts and handed the vendor's own knobs.

    The descriptor is ``data.source`` read as written. The square size and the batch come
    from the run rather than from a second declaration: ``image_size`` is what the
    transforms would have resized to, and ``loader.batch_size`` is the one the loader is
    built with, so neither can drift from what the run actually does.

    Which knobs are the vendor's own is decided by the model's signature rather than by a
    table of key names — what its constructor does not name is forwarded, so one
    declaration serves the model and the pipeline alike.
    """
    architecture = resolve_target(config.model, model_registry)
    ours = named_by(architecture, config.model.params)
    vendor = {name: value for name, value in config.model.params.items() if name not in ours}
    (task_name,) = config.tasks
    built: DataModule = YoloDataModule(
        data_yaml=_descriptor(config),
        task_name=task_name,
        image_size=config.image_size[0],
        batch_size=config.loader.batch_size,
        **vendor,
    )
    return built


def _descriptor(config: ExperimentConfig) -> str:
    """The one path a vendor descriptor is, refused by name when several were declared."""
    source = config.data.source
    # Duck-typed on the one field that answers the question, so this reads the shape it
    # needs without importing a config class at runtime for a single isinstance.
    declared = getattr(source, "path", None)
    if isinstance(declared, str):
        return declared
    raise ValueError(
        "A vendor family reads one descriptor naming its own stages, so 'data.source' has to "
        f"be a single path; got {type(source).__name__}."
    )
