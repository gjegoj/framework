"""Class counts inferred during data setup, consumed when tasks are built."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RuntimeContext:
    """Per-task class counts inferred from data, threaded from setup to task building.

    ``DataModule.setup()`` fits the target encoders and fills ``num_classes``;
    ``build_tasks()`` reads it to size each head.

    Values known only once the trainer is attached (total optimizer steps, epoch
    count) are deliberately **not** kept here — they are read directly from
    ``trainer`` at the lifecycle point that needs them (e.g.
    ``LitModule.configure_optimizers``, the EMA callback), mirroring the rest of
    the codebase.
    """

    num_classes: dict[str, int] = field(default_factory=dict)
