"""What a run distils from, and how strongly."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.config.components import ModelConfig
from src.config.tasks import LossConfig


class TeacherConfig(BaseModel):
    """One frozen teacher: an architecture, and optionally the weights to fill it with.

    Its heads are derived from the student's tasks, so the two models' logits
    match in shape by construction rather than by a user keeping two declarations
    in step.
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(
        description=(
            "The teacher's backbone, declared exactly as the student's 'model' section is — the same "
            "alias, so the two positions cannot drift into different shapes."
        )
    )
    checkpoint_path: str | None = Field(
        None,
        description=(
            "A run's checkpoint holding this teacher's weights. None keeps whatever the architecture "
            "was built with: a backbone declared 'pretrained: true' is already a teacher, and "
            "requiring a file would make that inexpressible."
        ),
    )


class DistillationConfig(BaseModel):
    """Training a student beside frozen teachers.

    Each task's training loss gains a soft term comparing the student's logits
    with the teachers' averaged ones. Nothing changes off training, so validation
    and test report the same quantity an undistilled run does.

    The soft term is declared as any other loss term is — a ``LossConfig``, weight
    included — because that is what it is: one more criterion inside a task's
    total, added with its weight and logged under its own name.

    A root section rather than a callback, because it changes what the model
    *computes*: no Lightning hook's return value reaches the loss, and a callback
    adding a backward pass of its own would contribute unscaled gradients against
    AMP's scaled ones. A technique that only changes what the model *holds* — EMA,
    freezing — is a callback.
    """

    model_config = ConfigDict(extra="forbid")

    teachers: list[TeacherConfig] = Field(
        min_length=1,
        description="Frozen teachers whose raw logits are averaged into one soft target.",
    )
    loss: LossConfig | list[LossConfig] = Field(
        default_factory=lambda: LossConfig(name="kl_divergence"),
        description=(
            "How the two models' logits are compared, and how strongly the comparison pulls beside "
            "the hard signal: {name: kl_divergence, temperature: 2.0, weight: 0.7}. The temperature "
            "lives here and nowhere else. A list declares several comparisons, added with their weights."
        ),
    )
