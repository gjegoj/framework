"""The root of the config contract: one validated object per experiment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.components import ComponentConfig, ModelConfig, TransformConfig
from src.config.data import DataConfig
from src.config.distillation import DistillationConfig
from src.config.run import RunConfig
from src.config.tasks import TaskConfig
from src.config.training import LoaderConfig, OptimizerConfig, SchedulerConfig, TrainerConfig
from src.core.normalisation import IMAGENET_MEAN, IMAGENET_STD
from src.core.taxonomy import Stage

CallbackConfig = ComponentConfig
"""One callback of the run: a registry name ('checkpoint', 'ema') or an import path, plus its arguments."""

LoggerConfig = ComponentConfig
"""The experiment tracker to build ('clearml' or an import path); None keeps Lightning's default."""

ExporterConfig = ComponentConfig
"""One deployment format to write ('torchscript') or an import path, plus its arguments."""

AdaptersConfig = ComponentConfig
"""A parameter-efficient technique to apply to the backbone ('lora'), plus its arguments."""


class ExperimentConfig(BaseModel):
    """The single validated source of truth for one experiment.

    Validation happens exactly once, at ``load_config``; everything downstream
    works with this object and never re-parses raw dicts. Structural sections
    reject unknown keys (typo protection); forward sections (loader, trainer)
    and components keep them as pass-through knobs.

    ``model`` is a plain component: one shape for every model family, with the
    family following from the name rather than a switch field. Heads are never
    configured — they are derived from tasks and sized from data facts.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(
        42,
        description=(
            "Everything stochastic inside a run: weight init, augmentation, batch order. The data "
            "split has its own seed ('data.split.seed') on purpose, so repeating a run at several "
            "seeds keeps one test set."
        ),
    )
    lr: float = Field(1.0e-3, gt=0, description="Shared learning rate; reach it with ${lr} wherever it belongs.")
    epochs: int = Field(10, gt=0, description="Shared epoch count; reach it with ${epochs}.")
    batch_size: int = Field(16, gt=0, description="Shared batch size; reach it with ${batch_size}.")
    image_size: tuple[int, int] = Field((224, 224), description="Shared (height, width); reach it with ${image_size}.")
    mean: list[float] = Field(
        default_factory=lambda: list(IMAGENET_MEAN),
        description="Shared per-channel normalisation mean; reach it with ${mean}.",
    )
    std: list[float] = Field(
        default_factory=lambda: list(IMAGENET_STD),
        description="Shared per-channel normalisation deviation; reach it with ${std}.",
    )

    data: DataConfig = Field(description="Where the annotation rows come from and how they feed the model.")
    tasks: dict[str, TaskConfig] = Field(
        description="Learned objectives by name; the name prefixes every loss and metric this task logs.",
    )
    model: ModelConfig = Field(
        description="The model to build: a registry name ('timm', 'smp') or an import path, plus its arguments.",
    )
    adapters: AdaptersConfig | None = Field(
        None,
        description=(
            "Train a small delta instead of the backbone's weights: {name: lora, target_modules: "
            "[qkv, proj], rank: 8}. The base freezes, the adapters and the heads learn, and the "
            "delta folds back before anything reads the weights. None trains every weight."
        ),
    )
    distillation: DistillationConfig | None = Field(
        None,
        description=(
            "Train beside frozen teachers: {teachers: [{model: {...}, checkpoint_path: ...}], loss: "
            "{name: kl_divergence, temperature: 2.0, weight: 0.7}}. Each task's training loss gains "
            "the declared term, comparing the student's logits with the teachers' averaged ones; "
            "evaluation is untouched. None trains against the data alone."
        ),
    )
    transforms: dict[Stage, TransformConfig] | None = Field(
        None,
        description="Per-stage sample pipeline; None leaves samples as the loaders produced them.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(name="adamw"),
        description="Optimizer by registry name or import path; its arguments forward to the constructor.",
    )
    scheduler: SchedulerConfig | None = Field(
        None,
        description="Learning-rate schedule plus how Lightning steps it; None keeps the rate fixed.",
    )
    loader: LoaderConfig = Field(default_factory=LoaderConfig, description="DataLoader knobs shared by every stage.")
    trainer: TrainerConfig = Field(default_factory=TrainerConfig, description="Lightning Trainer knobs.")
    callbacks: list[CallbackConfig] | None = Field(
        None,
        description=(
            "What the run does around its training steps, in the order given — keep the best "
            "checkpoint, log the learning rate, freeze a backbone. Order is not cosmetic: a "
            "callback that changes the weights belongs before one that saves them."
        ),
    )
    logger: LoggerConfig | None = Field(
        None,
        description=(
            "The experiment tracker ({name: clearml, project_name: ...}); constructor knobs "
            "forward verbatim. None keeps Lightning's default logging."
        ),
    )
    export: list[ExporterConfig] | None = Field(
        None,
        description=(
            "Deployment formats written after the run, in the order given; each entry names an "
            "exporter and carries that format's own knobs. None or an empty list writes nothing."
        ),
    )
    run: RunConfig = Field(default_factory=RunConfig, description="Which stages to run, from where, into where.")

    @model_validator(mode="after")
    def _require_named_tasks(self) -> ExperimentConfig:
        """A run without a task learns nothing, and a blank name has no key to log under."""
        if not self.tasks:
            raise ValueError("An experiment needs at least one task.")
        blank = [name for name in self.tasks if not name.strip()]
        if blank:
            raise ValueError("Task names must be non-empty.")
        return self
