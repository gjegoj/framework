"""The tasks section: one learned task per entry, keyed by name."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config.components import ComponentConfig, MetricConfig

HeadConfig = ComponentConfig
"""The head to build for a task: a registry name ('cosine') or an import path.

Sizes are never written here — ``in_features`` and ``out_features`` stay
derived from the backbone stream and the task's facts.
"""

TargetEncoderConfig = ComponentConfig
"""The encoder turning a target cell into a tensor: a registry name ('label', 'mask') or an import path."""

RETIRED_KEYS = ("preset", "output_topology", "input_topology", "objective")
"""The spellings a task used to be declared by; each is refused naming ``kind``."""


class LossConfig(ComponentConfig):
    """One criterion, alone or as a part of a composite loss.

    Inherits the component grammar (``name`` / ``_target_`` / params); ``weight``
    is declared, so it never leaks into the criterion's constructor arguments.
    One shape for both uses, so a loss keeps its weight when it moves in or out
    of a list.
    """

    weight: float = Field(1.0, gt=0, description="Multiplier of this criterion inside the task's total.")


class TaskConfig(BaseModel):
    """One task as declared in config.

    ``kind`` names what the task is — a familiar name (``classification``,
    ``segmentation``) or, for a kind of your own, the component form with ``_target_``.
    The kind states the defaults; the target column and its encoder are declared here,
    once — the data schema derives from tasks (single source of truth). ``None`` for
    ``target_encoder``, ``loss``, ``streams`` and ``metrics`` means "the kind's default",
    chosen at build time.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ComponentConfig = Field(
        description=(
            "What the task is: a familiar name — classification, binary_classification, "
            "multilabel_classification, regression, metric_learning, segmentation, binary_segmentation, "
            "multilabel_segmentation, contrastive, ranking, detection — or {_target_: my_pkg.Depth} for a kind of your own."
        ),
    )
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
            "How a target cell becomes a tensor. None takes the encoder the kind implies — "
            "class indices for classification, an indicator vector for multilabel, the value itself "
            "for regression — so declaring one is an override. Per-pixel targets are the exception: "
            "a mask is a file of its own and its encoder needs the class count."
        ),
    )
    loss: LossConfig | list[LossConfig] | None = Field(
        None,
        description=(
            "Criterion for this task; None takes the kind's default. A list declares several "
            "criteria on the same output, added with their weights and logged term by term."
        ),
    )
    head: HeadConfig | None = Field(
        None,
        description=(
            "Which head serves this task; None takes the kind's default (linear for a global output, "
            "conv for a dense one). Sizes are always derived, so an override names the kind only — "
            "{name: cosine} for an angular-margin classifier — or the reserved name 'native' for the "
            "head the backbone brings, which matters when those weights are the point."
        ),
    )
    streams: tuple[str, ...] | None = Field(
        None,
        description=(
            "Which backbone streams the head reads — one name or a list, in reading order. Absent, the "
            "kind's default ('features', 'decoder') or, for a detection task, the backbone's pyramid."
        ),
    )
    weight: float = Field(1.0, gt=0, description="Multiplier of this task's loss in the total.")
    lr: float | None = Field(
        None,
        gt=0,
        description=(
            "Own learning rate for this task's components — its head and its criterion. None shares "
            "the run's rate; the backbone always follows the optimizer section."
        ),
    )
    metrics: dict[str, MetricConfig] | None = Field(
        None,
        description=(
            "Metrics by the label they log under; every entry names its metric ('name' or "
            "'_target_'), so two flavours of one metric may stand side by side — "
            "{f1_macro: {name: f1, average: macro}}. None takes the kind's default set, whole; "
            "a declared mapping replaces it whole."
        ),
    )

    @field_validator("streams", mode="before")
    @classmethod
    def _one_name_or_several(cls, value: object) -> object:
        """``streams: encoder`` and ``streams: [p4, p5]`` are one key: a string is a one-tuple."""
        return (value,) if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_retired_spellings(cls, data: object) -> object:
        """A task is declared by its kind; the preset and the explicit axes it replaced are named, not guessed at."""
        if not isinstance(data, dict):
            return data
        retired = [key for key in RETIRED_KEYS if key in data]
        if retired:
            raise ValueError(
                f"Task keys {', '.join(retired)} are gone: declare the task's 'kind' instead "
                f"(kind: classification, or kind: {{_target_: my_pkg.Depth}})."
            )
        return data
