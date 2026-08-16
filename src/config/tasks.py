"""The tasks section: one declared objective per entry, keyed by task name."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.components import ComponentConfig, MetricConfig
from src.config.presets import resolve_preset
from src.core.taxonomy import InputTopology, Objective, OutputTopology
from src.core.vocabulary import ordered_names

HeadConfig = ComponentConfig
"""The head to build for a task: a registry name ('cosine') or an import path.

Sizes are never written here — ``in_features`` and ``out_features`` stay
derived from the backbone stream and the profiled data facts.
"""

TargetEncoderConfig = ComponentConfig
"""The encoder turning a target cell into a tensor: a registry name ('label', 'mask') or an import path."""


class LossConfig(ComponentConfig):
    """One criterion, alone or as a part of a composite loss.

    Inherits the component grammar (``name`` / ``_target_`` / params); ``weight``
    is declared, so it never leaks into the criterion's constructor arguments.
    One shape for both uses, so a loss keeps its weight when it moves in or out
    of a list.
    """

    weight: float = Field(1.0, gt=0, description="Multiplier of this criterion inside the task's total.")


class TaskConfig(BaseModel):
    """One learned objective as declared in config.

    Declare either a ``preset`` (a familiar name) or both explicit axes;
    a preset is resolved before validation, so ``output_topology`` and ``objective``
    are always concrete afterwards. The target column and its encoder are
    declared here, once — the data schema derives from tasks (single source
    of truth). ``None`` for ``target_encoder``, ``loss``, and ``stream`` means
    "the objective's or topology's default", chosen at assembly.
    """

    model_config = ConfigDict(extra="forbid")

    preset: str | None = Field(
        None,
        description="Familiar name ('classification', 'segmentation') standing for a point on the task axes.",
    )
    output_topology: OutputTopology = Field(description="Shape of the prediction: per-sample, per-pixel, per-object.")
    input_topology: InputTopology = Field(
        default=InputTopology.SINGLE,
        description="How the inputs are arranged: one per sample, N views, or separate streams.",
    )
    objective: Objective = Field(description="What is being learned: single-label, multi-label, regression.")
    target: str | None = Field(None, description="Table column holding this task's ground truth.")
    classes: dict[int, str] | None = Field(
        None,
        description=(
            "Declared class vocabulary, index to name ({0: cat, 1: dog}) — the source of truth "
            "the data is validated against, and the names logs and exports speak. None learns "
            "the vocabulary from the training split."
        ),
    )
    target_encoder: TargetEncoderConfig | None = Field(
        None,
        description=(
            "How a target cell becomes a tensor. None takes the encoder the objective implies — "
            "class indices for multiclass, an indicator vector for multilabel, the value itself "
            "for regression — so declaring one is an override. Per-pixel targets are the exception: "
            "a mask is a file of its own and its encoder needs the class count."
        ),
    )
    loss: LossConfig | list[LossConfig] | None = Field(
        None,
        description=(
            "Criterion for this task; None takes the objective's default. A list declares several "
            "criteria on the same output, added with their weights and logged term by term."
        ),
    )
    head: HeadConfig | None = Field(
        None,
        description=(
            "Which kind of head serves this task; None takes the topology's default (linear for "
            "global, conv for dense). Sizes are always derived, so an override names the kind only "
            "— e.g. {name: cosine} for an angular-margin classifier."
        ),
    )
    stream: str | None = Field(
        None,
        description="Backbone output this task's head reads ('features', 'encoder'); None takes the topology's default.",
    )
    weight: float = Field(1.0, gt=0, description="Multiplier of this task's loss in the total.")
    lr: float | None = Field(
        None,
        gt=0,
        description=(
            "Own learning rate for this task's bricks — its head and its criterion. None shares "
            "the run's rate; the backbone always follows the optimizer section."
        ),
    )
    metrics: dict[str, MetricConfig] | None = Field(
        None,
        description=(
            "Metrics by the label they log under; every entry names its metric ('name' or "
            "'_target_'), so two flavours of one metric may stand side by side — "
            "{f1_macro: {name: f1, average: macro}}. None takes the objective's default set."
        ),
    )
    native_head: bool = Field(
        False,
        description=(
            "Keep the head the pretrained model ships with instead of deriving one from the task. "
            "Needed when those weights are the point — a detector, a released classifier."
        ),
    )

    @model_validator(mode="after")
    def _one_way_of_choosing_a_head(self) -> TaskConfig:
        """Both keys answer the same question, and together one of them would win silently."""
        if self.head is not None and self.native_head:
            raise ValueError(
                "Set either 'head' (build this kind) or 'native_head' (keep the backbone's own), not both."
            )
        return self

    @model_validator(mode="after")
    def _classes_form_a_contract(self) -> TaskConfig:
        """The declared vocabulary *is* the index space, so it has to be complete and unambiguous."""
        if self.classes is None:
            return self
        if self.objective is Objective.CONTINUOUS:
            raise ValueError("'classes' declared for a continuous objective; bins own its value space.")
        # Called for the refusal, not the list: config keeps the mapping it was given, and
        # the ordered names are what an encoder wants. The rule itself has one owner, so
        # the message a user sees is the same whether they declared the vocabulary on a
        # task or handed it to an encoder from Python.
        ordered_names(self.classes)
        return self

    @model_validator(mode="before")
    @classmethod
    def _resolve_the_preset(cls, data: object) -> object:
        """Expand a familiar name into axes and customary metrics, before anything reads them."""
        if not isinstance(data, dict) or data.get("preset") is None:
            return data
        if "output_topology" in data or "input_topology" in data or "objective" in data:
            raise ValueError(
                "Set either 'preset' or explicit 'output_topology'/'input_topology'/'objective', not both."
            )
        try:
            preset = resolve_preset(data["preset"])
        except LookupError as error:
            # Pydantic wraps only ValueError into ValidationError; keep the message.
            raise ValueError(str(error)) from error
        resolved = {
            **data,
            "output_topology": preset.output_topology,
            "input_topology": preset.input_topology,
            "objective": preset.objective,
        }
        if preset.metrics is not None and "metrics" not in data:
            # The kind's customary judgment, injected where the user said nothing —
            # visible in the loaded config and validated by the same grammar.
            resolved["metrics"] = preset.metrics
        return resolved
