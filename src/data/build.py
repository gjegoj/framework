"""The data section and the preprocessing section, turned into a preprocessor and a data module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import ClassFile, ComponentConfig, PreprocessingConfig, TaskConfig
from src.config.instantiate import instantiate, resolve_params, resolve_target
from src.core import Stage
from src.data.base import DataModule, Encoder, Preprocessor, TableSource, TargetEncoder
from src.data.registry import (
    cache_registry,
    collator_registry,
    data_module_registry,
    input_encoder_registry,
    preprocessor_registry,
    target_encoder_registry,
)
from src.data.sources import source_for
from src.data.split import Split
from src.data.table import TableDataModule
from src.transforms import GeometryAware, SampleTransform


def build_preprocessor(
    declared: PreprocessingConfig | None,
    tasks: Mapping[str, TaskConfig],
    default_encoders: Mapping[str, ComponentConfig | None],
) -> Preprocessor:
    """Input encoders from the preprocessing section, target encoders from the tasks, one collator."""
    if declared is None:
        raise ValueError("preprocessing is required: declare how each input becomes a tensor (preprocessing=image).")
    targets = {
        name: build_target_encoder(name, task, default_encoders.get(name))
        for name, task in tasks.items()
        if task.target
    }
    built: Preprocessor = instantiate(
        declared,
        preprocessor_registry,
        inputs=_encoders(declared.inputs, input_encoder_registry),
        auxiliary_inputs=_encoders(declared.auxiliary_inputs, input_encoder_registry),
        targets=targets,
        collator=instantiate(declared.collator or ComponentConfig(name="stack"), collator_registry),
        cache=instantiate(declared.cache, cache_registry) if declared.cache is not None else None,
    )
    return built


def build_target_encoder(name: str, task: TaskConfig, default: ComponentConfig | None) -> TargetEncoder:
    """The declared encoder, or the task kind's default; ``classes`` come from the task and nowhere else."""
    component = task.target_encoder or default
    if component is None:
        raise ValueError(f"Task {name!r} declares a target but no target_encoder, and its kind has no default.")
    if "classes" in component.params:
        raise ValueError(f"Task {name!r}: classes are declared on the task, not inside its target encoder.")
    factory = resolve_target(component, target_encoder_registry)
    reads_classes = isinstance(factory, type) and issubclass(factory, Encoder) and factory.takes_classes
    if reads_classes:
        if task.classes is None:
            raise ValueError(f"Task {name!r}: {component.spelled!r} reads a vocabulary; declare classes on the task.")
        encoder: TargetEncoder = instantiate(component, target_encoder_registry, classes=classes_of(task.classes))
        return encoder
    if task.classes is not None:
        raise ValueError(f"Task {name!r} declares classes, but {component.spelled!r} reads no vocabulary; drop them.")
    encoder = instantiate(component, target_encoder_registry)
    return encoder


def classes_of(declared: Mapping[int, str] | ClassFile) -> dict[int, str]:
    """A vocabulary as declared, or one name per line of the named file."""
    if isinstance(declared, ClassFile):
        names = [line.strip() for line in Path(declared.file).read_text(encoding="utf-8").splitlines() if line.strip()]
        return dict(enumerate(names))
    return dict(declared)


def build_transforms(
    declared: Mapping[Stage, ComponentConfig], geometries: Mapping[str, Mapping[str, Any]]
) -> dict[str, SampleTransform]:
    """One sample transform per split name; a transform that moves pixels is told what moves with them."""
    built: dict[str, SampleTransform] = {}
    for stage, component in declared.items():
        transform = instantiate(component)
        built[stage.value] = (
            transform.with_geometry(**geometries) if isinstance(transform, GeometryAware) else transform
        )
    return built


def build_data_module(
    declared: ComponentConfig,
    *,
    preprocessor: Preprocessor,
    targets: Mapping[str, str],
    transforms: Mapping[str, SampleTransform],
) -> DataModule:
    """The data section as a module; the table module's declaration grammar is translated here."""
    factory = resolve_target(declared, data_module_registry)
    params = resolve_params(declared)
    if isinstance(factory, type) and issubclass(factory, TableDataModule):
        params = table_arguments(params)
    built: DataModule = factory(**params, preprocessor=preprocessor, targets=targets, transforms=transforms)
    return built


def table_arguments(params: Mapping[str, Any]) -> dict[str, Any]:
    """``source`` paths become sources, ``inputs`` bindings become columns, ``split`` becomes a Split."""
    translated = dict(params)
    if "source" in translated:
        translated["source"] = _sources(translated["source"])
    if "inputs" in translated:
        translated["inputs"] = {name: _column(name, binding) for name, binding in translated["inputs"].items()}
    if isinstance(translated.get("split"), Mapping):
        translated["split"] = Split(**translated["split"])
    return translated


def _sources(declared: Any) -> TableSource | dict[str, TableSource]:
    if isinstance(declared, Mapping) and "path" not in declared:
        return {str(name): _source(entry) for name, entry in declared.items()}
    return _source(declared)


def _source(declared: Any) -> TableSource:
    if isinstance(declared, Mapping):
        return source_for(declared["path"], format=declared.get("format"))
    return source_for(declared)


def _column(name: str, binding: Any) -> str:
    if isinstance(binding, Mapping):
        return str(binding["column"])
    if isinstance(binding, str):
        return binding
    raise ValueError(f"Input {name!r} binds to a column name or {{column: ...}}, got {binding!r}.")


def _encoders(declared: Mapping[str, ComponentConfig] | None, registry: Any) -> dict[str, Any]:
    return {name: instantiate(component, registry) for name, component in (declared or {}).items()}
