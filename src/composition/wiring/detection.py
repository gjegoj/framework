"""Detection assembler: native-YOLO runs behind the ``ExperimentAssembler`` port.

Detection diverges from the standard chain before data building (native YOLO data
format, no Task machinery). Capability flags own the v1 boundaries — export, LoRA,
distillation and task mixing are phase-2 landing points documented in the
complete-models design spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.composition.wiring.common import forward_extras
from src.composition.wiring.experiment import Capabilities, ExperimentAssembler, experiment_assemblers
from src.composition.wiring.training import build_optimizer_builder, build_scheduler_builder
from src.data.detection import DetectionDataModule
from src.metrics.bundle import DETECTION_DEFAULT_METRICS, build_metric_bundle
from src.models.yolo import YoloModel
from src.training.modules import CompleteModelLitModule

if TYPE_CHECKING:
    import lightning as L

    from src.config.schema import ExperimentConfig
    from src.core.entities import Task
    from src.core.runtime import RuntimeContext


class DetectionExperimentAssembler(ExperimentAssembler):
    """Native-YOLO detection runs (see the complete-models design spec)."""

    name = "detection"
    capabilities = Capabilities(export=False, lora=False, distillation=False, batch_transforms=False, task_mixing=False)

    def validate(self, config: ExperimentConfig) -> None:
        """Capability checks plus the detection-specific model requirement.

        Parameters:
            config (ExperimentConfig): Validated experiment config with a detection task.

        Raises:
            ValueError: Outside the v1 boundaries or without a model to build.
        """
        super().validate(config)
        if config.model.name is None:
            raise ValueError(
                "Detection needs the model section's 'name': an ultralytics .yaml architecture or .pt weights path "
                "(e.g. 'model: {kind: yolo, name: yolov8n.yaml}')."
            )

    def build(
        self, config: ExperimentConfig, runtime: RuntimeContext
    ) -> tuple[L.LightningModule, L.LightningDataModule, list[Task]]:
        """Assemble the detection run: YOLO data module + fused model + generic module.

        Parameters:
            config (ExperimentConfig): Validated experiment config with a detection task.
            runtime (RuntimeContext): Receives ``num_classes`` from the dataset descriptor.

        Returns:
            tuple[L.LightningModule, L.LightningDataModule, list[Task]]: Ready for
            the shared trainer tail; the task list is empty (no Task machinery).
        """
        task_name = _detection_task_name(config)
        hyperparameters = forward_extras(config.model, _MODEL_SECTION_CORE_FIELDS)
        datamodule = DetectionDataModule(
            data_yaml=_data_yaml_path(config),
            image_size=config.image_size[0],
            batch_size=config.batch_size,
            hyperparameters=hyperparameters,
            num_workers=config.dataloader.num_workers,
        )
        datamodule.setup()
        runtime.num_classes[task_name] = datamodule.num_classes
        model = YoloModel(
            name=_require_model_name(config),
            num_classes=runtime.num_classes[task_name],
            image_size=config.image_size[0],
            hyperparameters=hyperparameters,
        )
        lit_module = CompleteModelLitModule(
            model=model,
            task_name=task_name,
            metric_bundle=build_metric_bundle(config.tasks[task_name].metrics, DETECTION_DEFAULT_METRICS),
            optimizer_builder=build_optimizer_builder(config.optimizer),
            scheduler_builder=build_scheduler_builder(config.scheduler),
            hparams=config.model_dump(mode="json"),
        )
        return lit_module, datamodule, []


def _detection_task_name(config: ExperimentConfig) -> str:
    """The single detection task's name (guaranteed by the capability validation)."""
    for name, task in config.tasks.items():
        if task.preset == DetectionExperimentAssembler.name:
            return name
    raise ValueError("No detection task configured — resolve_experiment_assembler dispatched incorrectly.")


def _data_yaml_path(config: ExperimentConfig) -> str:
    """The YOLO ``data.yaml`` descriptor path from ``data.sources``."""
    sources = config.data.sources
    if not isinstance(sources, str):
        raise ValueError(  # noqa: TRY004 — a user-config shape error, not a Python-type contract violation
            f"Detection expects data.sources to be a single YOLO data.yaml path, got {type(sources).__name__}."
        )
    return sources


_MODEL_SECTION_CORE_FIELDS = frozenset({"kind", "name", "pretrained"})


def _require_model_name(config: ExperimentConfig) -> str:
    if config.model.name is None:
        raise ValueError("Detection needs the model section's 'name': an ultralytics .yaml architecture or .pt path.")
    return config.model.name


experiment_assemblers.register_instance(DetectionExperimentAssembler.name, DetectionExperimentAssembler())
