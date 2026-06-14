"""SchedulerBuilder: resolve a torch LR scheduler from config and bind it to an optimizer.

Mirrors :class:`OptimizerBuilder`. Schedulers are third-party ``torch.optim.lr_scheduler``
classes selected from the ``schedulers`` registry by name, with kwargs forwarded
verbatim. Values known only once the trainer is attached (``total_steps``, …) are read
from the trainer in ``configure_optimizers`` and passed to ``build`` as ``trainer_facts``;
``runtime_kwargs`` (from config) declares which constructor parameter receives which fact,
so ``build`` itself stays scheduler-agnostic.

``build`` returns Lightning's ``lr_scheduler`` config dict; the humble ``LitModule`` hands
it straight to Lightning.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lightning.pytorch.utilities.types import LRSchedulerConfigType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.training.registry import schedulers

# Trainer-derived facts a scheduler may request, mapped to the attribute that supplies each.
# Single source of truth: read in configure_optimizers, validated in build_scheduler_builder.
TRAINER_FACTS: dict[str, str] = {
    "total_steps": "estimated_stepping_batches",
    "steps_per_epoch": "num_training_batches",
    "epochs": "max_epochs",
}


class SchedulerBuilder:
    """Builds a torch LR scheduler and its Lightning scheduling policy.

    Parameters:
        scheduler_cls (Callable[..., LRScheduler]): Scheduler class (registry value).
        name (str): Scheduler key, surfaced to Lightning's LR monitor.
        interval (str): ``"epoch"`` or ``"step"`` — Lightning step cadence.
        frequency (int): Step every N intervals.
        monitor (str | None): Metric to monitor (ReduceLROnPlateau); omitted when None.
        strict (bool): Error if the monitored metric is missing.
        runtime_kwargs (dict[str, str] | None): Constructor param → trainer-fact name.
        extra_kwargs (dict[str, Any] | None): Static constructor kwargs from config.
    """

    def __init__(
        self,
        scheduler_cls: Callable[..., LRScheduler],
        name: str,
        *,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str | None = None,
        strict: bool = True,
        runtime_kwargs: dict[str, str] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._scheduler_cls = scheduler_cls
        self._name = name
        self._interval = interval
        self._frequency = frequency
        self._monitor = monitor
        self._strict = strict
        self._runtime_kwargs = runtime_kwargs or {}
        self._extra_kwargs = extra_kwargs or {}

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        interval: str = "epoch",
        frequency: int = 1,
        monitor: str | None = None,
        strict: bool = True,
        runtime_kwargs: dict[str, str] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> SchedulerBuilder:
        """Build from a scheduler registry key (e.g. ``"cosine"``/``"onecycle"``)."""
        return cls(
            schedulers.get(name),
            name,
            interval=interval,
            frequency=frequency,
            monitor=monitor,
            strict=strict,
            runtime_kwargs=runtime_kwargs,
            extra_kwargs=extra_kwargs,
        )

    def build(self, optimizer: Optimizer, trainer_facts: dict[str, int]) -> LRSchedulerConfigType:
        """Construct the scheduler and return Lightning's typed ``lr_scheduler`` config.

        Parameters:
            optimizer (Optimizer): The already-built optimizer to schedule.
            trainer_facts (dict[str, int]): Trainer-derived values (see ``TRAINER_FACTS``)
                routed into constructor kwargs via ``runtime_kwargs``.

        Returns:
            LRSchedulerConfigType: ``{"scheduler", "name", "interval", "frequency", "strict"}``
                plus ``"monitor"`` when one is configured.
        """
        kwargs = dict(self._extra_kwargs)
        for param, fact in self._runtime_kwargs.items():
            kwargs[param] = trainer_facts[fact]
        scheduler = self._scheduler_cls(optimizer, **kwargs)
        config: LRSchedulerConfigType = {
            "scheduler": scheduler,
            "name": self._name,
            "interval": self._interval,
            "frequency": self._frequency,
            "strict": self._strict,
        }
        if self._monitor is not None:
            config["monitor"] = self._monitor
        return config
