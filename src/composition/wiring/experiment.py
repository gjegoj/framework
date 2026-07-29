"""Experiment assemblers: one strategy per run family, resolved from config.

``main.py`` is a Front Controller: resolve an assembler, validate, build, hand the
pair to the shared trainer tail. The standard topology-x-objective chain is itself an
assembler — the default member of the same family — so adding a run family never
touches ``main.py`` (the ``callback_builders`` convention, applied to the whole run).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from src.composition.wiring.data import build_data_module
from src.composition.wiring.export import validate_export_preconditions
from src.composition.wiring.model import apply_lora_if_configured, build_backbone, validate_lora_preconditions
from src.composition.wiring.tasks import build_bindings, build_tasks
from src.composition.wiring.training import (
    build_lit_data_module,
    build_lit_module,
    build_optimizer_builder,
    build_scheduler_builder,
    build_task_lr_overrides,
)
from src.core.registry import Registry
from src.models.assembly import build_composite_model

if TYPE_CHECKING:
    import lightning as L

    from src.config.schema import ExperimentConfig
    from src.core.entities import Task
    from src.core.runtime import RuntimeContext


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a run family supports; the shared guard turns ``False`` into fail-fast errors.

    Parameters:
        export (bool): ``run_export: true`` allowed.
        lora (bool): A ``lora:`` section allowed.
        distillation (bool): A ``distillation:`` section allowed.
        batch_transforms (bool): Batch-transform callbacks allowed.
        task_mixing (bool): More than one task allowed.
    """

    export: bool
    lora: bool
    distillation: bool
    batch_transforms: bool
    task_mixing: bool


FULL_CAPABILITIES = Capabilities(export=True, lora=True, distillation=True, batch_transforms=True, task_mixing=True)


class ExperimentAssembler(ABC):
    """Builds a run family: validation up front, then the Lightning pair + tasks."""

    name: ClassVar[str]
    capabilities: ClassVar[Capabilities]

    def validate(self, config: ExperimentConfig) -> None:
        """Fail fast on anything the family does not support (before building).

        Parameters:
            config (ExperimentConfig): Validated experiment config.

        Raises:
            ValueError: On a config outside the family's capabilities.
        """
        family = self.name
        if config.run_export and not self.capabilities.export:
            raise ValueError(f"{family.capitalize()} export is phase 2 — set run_export: false for {family} runs.")
        if config.lora is not None and not self.capabilities.lora:
            raise ValueError(f"LoRA is phase 2 for {family} runs — remove the lora: section.")
        if config.distillation is not None and not self.capabilities.distillation:
            raise ValueError(f"{family.capitalize()} distillation is phase 2 — remove the distillation: section.")
        if len(config.tasks) > 1 and not self.capabilities.task_mixing:
            raise ValueError(
                f"{family.capitalize()} runs support exactly one single {family} task and no other tasks "
                f"(mixing needs the shared-backbone Option-A work). Configured tasks: {sorted(config.tasks)}."
            )

    @abstractmethod
    def build(
        self, config: ExperimentConfig, runtime: RuntimeContext
    ) -> tuple[L.LightningModule, L.LightningDataModule, list[Task]]:
        """Assemble the run: Lightning module + datamodule + tasks (empty when fused).

        Parameters:
            config (ExperimentConfig): Validated experiment config.
            runtime (RuntimeContext): Shared runtime facts (``num_classes``), filled here.

        Returns:
            tuple[L.LightningModule, L.LightningDataModule, list[Task]]: Ready for the
            shared trainer tail (``build_trainer`` -> ``run_experiment``).
        """


experiment_assemblers: Registry[ExperimentAssembler] = Registry("experiment assembler")


def resolve_experiment_assembler(config: ExperimentConfig) -> ExperimentAssembler:
    """Pick the assembler by the model section's ``kind``; the standard chain is the default.

    A ``kind`` registered in ``complete_models`` selects that model's family assembler
    and requires every task to carry the family preset; a family preset without such a
    ``kind`` is rejected symmetrically, so the two declarations cannot drift apart.

    Parameters:
        config (ExperimentConfig): Validated experiment config.

    Returns:
        ExperimentAssembler: The registered family assembler, or the standard one.

    Raises:
        ValueError: On an incoherent kind/preset pairing.
    """
    from src.models.complete import CompleteModel
    from src.models.registry import complete_models

    kind = config.model.kind
    presets = {task.preset for task in config.tasks.values()}
    if kind in complete_models:
        family = cast("type[CompleteModel[Any, Any]]", complete_models.get(kind)).family
        if presets != {family}:
            raise ValueError(
                f"model kind {kind!r} belongs to the {family!r} family — every task must use preset "
                f"{family!r} (got {sorted(presets)})."
            )
        return experiment_assemblers.create(family)
    family_presets = presets & set(experiment_assemblers.keys()) - {StandardExperimentAssembler.name}
    if family_presets:
        raise ValueError(
            f"task preset(s) {sorted(family_presets)} need a complete-model kind in the model section "
            f"(e.g. 'model: {{kind: yolo, name: yolov8n.yaml}}'), got kind {kind!r}."
        )
    return experiment_assemblers.create(StandardExperimentAssembler.name)


class StandardExperimentAssembler(ExperimentAssembler):
    """The topology-x-objective chain (bindings data -> tasks -> composite model)."""

    name = "standard"
    capabilities = FULL_CAPABILITIES

    def validate(self, config: ExperimentConfig) -> None:
        super().validate(config)
        validate_lora_preconditions(config)
        # Export preconditions need built tasks — checked inside build(), before training.

    def build(
        self, config: ExperimentConfig, runtime: RuntimeContext
    ) -> tuple[L.LightningModule, L.LightningDataModule, list[Task]]:
        # 1. Data: read -> fit encoders (infers num_classes) -> split -> datasets
        bindings = build_bindings(config)
        plain_data_module = build_data_module(config, bindings, runtime)
        plain_data_module.setup()

        # 2. Tasks — built after setup so num_classes is a concrete int
        tasks = build_tasks(config, runtime)
        validate_export_preconditions(config, tasks)  # fail before training if export is impossible

        # 3. Model — heads sized from backbone.feature_dim, derived from tasks;
        #    LoRA (when configured) wraps the backbone and freezes its base weights
        backbone = build_backbone(config.model)
        model = build_composite_model(backbone, {task.name: task.head_spec for task in tasks})
        apply_lora_if_configured(config, model)

        # 4. Optimizer — per-head LR overrides (from task configs) bound into the builder
        optimizer_builder = build_optimizer_builder(config.optimizer, build_task_lr_overrides(config))
        scheduler_builder = build_scheduler_builder(config.scheduler)

        # 5. Lightning wrappers (humble objects delegating to domain logic)
        lit_module = build_lit_module(config, model, tasks, optimizer_builder, scheduler_builder)
        return lit_module, build_lit_data_module(plain_data_module), tasks


experiment_assemblers.register_instance(StandardExperimentAssembler.name, StandardExperimentAssembler())
