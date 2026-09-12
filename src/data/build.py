"""The data section and the preprocessing section, turned into a preprocessor and a data module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.config import ClassFile, ComponentConfig, PreprocessingConfig, TaskConfig
from src.config.instantiate import instantiate, resolve_factory
from src.core import Registry
from src.data.base import DataModule, Encoder, Preprocessor, TargetEncoder
from src.data.registry import (
    cache_registry,
    collator_registry,
    data_module_registry,
    input_encoder_registry,
    preprocessor_registry,
    target_encoder_registry,
)
from src.transforms import SampleTransform


def build_preprocessor(
    declared: PreprocessingConfig,
    tasks: Mapping[str, TaskConfig],
    default_encoders: Mapping[str, ComponentConfig | None],
) -> Preprocessor:
    """Input encoders from the preprocessing section, target encoders from the tasks, one collator."""
    targets = {
        name: build_target_encoder(name, task, default_encoders.get(name))
        for name, task in tasks.items()
        if task.target_column
    }
    built = instantiate(
        declared,
        preprocessor_registry,
        inputs=_encoders(declared.inputs, input_encoder_registry),
        auxiliary_inputs=_encoders(declared.auxiliary_inputs, input_encoder_registry),
        targets=targets,
        collator=instantiate(declared.collator or ComponentConfig(name="stack"), collator_registry),
        cache=instantiate(declared.cache, cache_registry) if declared.cache is not None else None,
    )
    if not isinstance(built, Preprocessor):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a Preprocessor: a run asks it "
            "to turn a raw sample into tensors and to join a batch of them, and this answers neither."
        )
    return built


def build_target_encoder(name: str, task: TaskConfig, default: ComponentConfig | None) -> TargetEncoder:
    """The declared encoder, or the task kind's default; ``classes`` come from the task and nowhere else."""
    component = task.target_encoder or default
    if component is None:
        raise ValueError(f"Task {name!r} declares a target but no target_encoder, and its kind has no default.")
    if "classes" in component.params:
        raise ValueError(f"Task {name!r}: classes are declared on the task, not inside its target encoder.")
    factory = resolve_factory(component, target_encoder_registry)
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


def build_data_module(
    declared: ComponentConfig,
    *,
    preprocessor: Preprocessor,
    targets: Mapping[str, str],
    transforms: Mapping[str, SampleTransform],
) -> DataModule:
    """The data section as a module, plus the three things every module is given rather than declares.

    A family with a declaration grammar of its own reads it in its own constructor — paths become
    sources, bindings become columns — so this builder knows of no family in particular.
    """
    built = instantiate(
        declared, data_module_registry, preprocessor=preprocessor, targets=targets, transforms=transforms
    )
    if not isinstance(built, DataModule):
        raise TypeError(
            f"{declared.spelled!r} built {type(built).__name__}, which is not a DataModule: a run asks it for "
            "prepared splits and what they turned out to hold, and this answers neither."
        )
    return built


def _encoders[T](declared: Mapping[str, ComponentConfig] | None, registry: Registry[T]) -> dict[str, T]:
    return {name: instantiate(component, registry) for name, component in (declared or {}).items()}
