"""Recognising a model family that arrives whole, and refusing what it cannot serve."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.registry import model_registry

if TYPE_CHECKING:
    from src.config import ExperimentConfig

_UNSERVICEABLE: tuple[tuple[str, str], ...] = (
    (
        "transforms",
        (
            "A vendor family augments through its own pipeline, which is box-aware; the "
            "'transforms' section is not. Move these knobs to the model section (mosaic, hsv_h, "
            "degrees) or to the dataset's own descriptor."
        ),
    ),
    (
        "export",
        (
            "A vendor family's output is not the per-task logits an exported graph is traced "
            "from. Drop the 'export' section for this run."
        ),
    ),
    (
        "adapters",
        (
            "Adapters reparameterize a backbone this framework composed, and a vendor family "
            "brings its own. Drop the 'adapters' section for this run."
        ),
    ),
    (
        "distillation",
        (
            "Distillation compares per-task logits, which a vendor family does not expose. "
            "Drop the 'distillation' section for this run."
        ),
    ),
)
"""Each section a vendor family cannot serve, beside the sentence that says why.

A table of messages rather than of booleans: the reason belongs with the refusal, and a
section added to the framework is refused here or not at all — there is no second list
to keep in step.

The sentences are written for the family that is shipped, and at least one of them is a
fact about *it* rather than about vendors in general — a box-aware pipeline is YOLO's
answer, not a law. So each refusal names the model it came from, and the reader knows
whose rule they hit. Making the table itself per-family is worth doing when there is a
second family to disagree with the first, and not before.
"""

_UNSERVICEABLE_CALLBACK = "batch_transform"
"""The one callback that rewrites targets, which only a tensor target can be."""


def is_vendor_family(config: ExperimentConfig) -> bool:
    """Whether this run's model brings its own head, loss and decoding.

    One name decides it: ``config.model.name`` found in ``model_registry`` rather than
    in ``backbone_registry``. Read from the model section and nowhere else, so the model
    and the data pipeline cannot disagree about which kind of run is being assembled.
    """
    return config.model.name is not None and config.model.name in model_registry


def refuse_what_a_vendor_cannot_serve(config: ExperimentConfig) -> None:
    """Fail before the first epoch, naming the section and where its knowledge belongs.

    A vendor family owns its head, its loss and its decoding, so most of what an
    experiment can declare has nothing to attach to: an augmentation pipeline that is
    not box-aware, an exporter with no logits to trace, an adapter with no backbone of
    ours to reparameterize.

    A section silently ignored is worse than a run that dies: it reports numbers for a
    recipe nobody ran, and the difference only shows up when someone tries to reproduce
    it. Refusing them from one function that reads the config means a section added to
    the framework is refused here or not at all — there is no second list to keep in step.
    """
    if not is_vendor_family(config):
        return
    family = str(config.model.name)
    for section, why in _UNSERVICEABLE:
        if getattr(config, section, None):
            raise ValueError(f"'{family}' cannot serve the '{section}' section. {why}")
    declared = [entry.name for entry in config.callbacks or []]
    if _UNSERVICEABLE_CALLBACK in declared:
        raise ValueError(
            f"The '{_UNSERVICEABLE_CALLBACK}' callback blends targets, and '{family}' targets "
            f"are objects rather than tensors. Drop it for this run."
        )
    _refuse_more_than_one_task(config, family)
    _refuse_our_bricks_on_its_task(config, family)


def _refuse_more_than_one_task(config: ExperimentConfig, family: str) -> None:
    """Its head is built for one task; a second would train nothing and report nothing."""
    if len(config.tasks) > 1:
        declared = ", ".join(sorted(config.tasks))
        raise ValueError(
            f"'{family}' serves one task, and this run declares {len(config.tasks)}: {declared}. "
            f"Its head is built for one, so the others would train nothing and report nothing."
        )


def _refuse_our_bricks_on_its_task(config: ExperimentConfig, family: str) -> None:
    """A head, a criterion or an encoder declared for a task whose family builds its own."""
    for name, task in config.tasks.items():
        declared = [field for field in ("head", "loss", "target_encoder") if getattr(task, field, None) is not None]
        if declared:
            raise ValueError(
                f"Task '{name}' declares {', '.join(declared)}, but '{family}' builds its own — "
                f"its assigner, its loss and its decoding are one design. Remove them from the task."
            )
