"""Detection wiring: routing, fail-fast guards, and experiment assembly.

Detection diverges from the standard chain before data building (native YOLO data
format, no Task machinery), so ``main.py`` branches early on ``is_detection_run``.
The guards keep the v1 boundaries explicit: one detection task per run, and no
export/LoRA/distillation (each is a documented phase-2 landing point in the spec).
"""

from __future__ import annotations

from src.composition.wiring.training import build_optimizer_builder, build_scheduler_builder
from src.config.schema import ExperimentConfig, TaskConfig
from src.data.detection import DetectionDataModule
from src.models.yolo import build_yolo_model
from src.training.modules import DetectionLitModule

DETECTION_PRESET = "detection"


def is_detection_run(config: ExperimentConfig) -> bool:
    """Whether any configured task uses the detection preset."""
    return any(task.preset == DETECTION_PRESET for task in config.tasks.values())


def validate_detection_preconditions(config: ExperimentConfig) -> None:
    """Reject configs outside the v1 detection boundaries, before anything is built.

    Parameters:
        config (ExperimentConfig): Validated experiment config with a detection task.

    Raises:
        ValueError: On task mixing, a missing ``model``, or an unsupported subsystem
            (export / LoRA / distillation — phase-2 landing points).
    """
    detection_tasks = {name: task for name, task in config.tasks.items() if task.preset == DETECTION_PRESET}
    if len(detection_tasks) != len(config.tasks) or len(detection_tasks) != 1:
        raise ValueError(
            "Detection runs support exactly one single detection task and no other tasks "
            "(mixing needs the shared-backbone Option-A work). "
            f"Configured tasks: {sorted(config.tasks)}."
        )
    task_name, task_config = next(iter(detection_tasks.items()))
    if task_config.model is None:
        raise ValueError(f"Detection task {task_name!r} needs 'model': an ultralytics .yaml architecture or .pt path.")
    if config.run_export:
        raise ValueError("Detection export is phase 2 — set run_export: false for detection runs.")
    if config.lora is not None:
        raise ValueError("LoRA on YOLO backbones is phase 2 — remove the lora: section for detection runs.")
    if config.distillation is not None:
        raise ValueError("Detection distillation is phase 2 — remove the distillation: section for detection runs.")


def build_detection_experiment(config: ExperimentConfig) -> tuple[DetectionLitModule, DetectionDataModule]:
    """Assemble the detection regime: data module (YOLO format) + Lightning module.

    Parameters:
        config (ExperimentConfig): Validated experiment config with a detection task.

    Returns:
        tuple[DetectionLitModule, DetectionDataModule]: Ready for ``L.Trainer.fit``.
    """
    task_name, task_config = _detection_task(config)
    datamodule = DetectionDataModule(
        data_yaml=_data_yaml_path(config),
        image_size=config.image_size[0],
        batch_size=config.batch_size,
        hyperparameters=task_config.hyperparameters,
        num_workers=config.dataloader.num_workers,
    )
    datamodule.setup()
    model = build_yolo_model(
        model=_require_model(task_name, task_config),
        num_classes=datamodule.num_classes,
        hyperparameters=task_config.hyperparameters,
    )
    lit_module = DetectionLitModule(
        model=model,
        task_name=task_name,
        image_size=config.image_size[0],
        optimizer_builder=build_optimizer_builder(config.optimizer),
        scheduler_builder=build_scheduler_builder(config.scheduler),
        hparams=config.model_dump(mode="json"),
    )
    return lit_module, datamodule


def _detection_task(config: ExperimentConfig) -> tuple[str, TaskConfig]:
    """The single detection task (guaranteed by ``validate_detection_preconditions``)."""
    for name, task in config.tasks.items():
        if task.preset == DETECTION_PRESET:
            return name, task
    raise ValueError("No detection task configured — call is_detection_run before building.")


def _data_yaml_path(config: ExperimentConfig) -> str:
    """The YOLO ``data.yaml`` descriptor path from ``data.sources``."""
    sources = config.data.sources
    if not isinstance(sources, str):
        raise TypeError(
            f"Detection expects data.sources to be a single YOLO data.yaml path, got {type(sources).__name__}."
        )
    return sources


def _require_model(task_name: str, task_config: TaskConfig) -> str:
    if task_config.model is None:
        raise ValueError(f"Detection task {task_name!r} needs 'model': an ultralytics .yaml architecture or .pt path.")
    return task_config.model
