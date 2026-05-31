"""OptimizerBuilder: assembles param-groups for per-head learning rates.

Per-head LR is a first-class feature (G1 in the plan). The builder separates
backbone and per-task head parameters into independent groups so each can have
its own lr/weight-decay without duplicating optimizer instances.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from src.training.registry import optimizers


@dataclass
class ParamGroupSpec:
    """Specification for one optimizer param-group.

    Parameters:
        name (str): Human-readable label (used by LR-monitor callbacks).
        lr (float): Learning rate for this group.
        weight_decay (float): Weight decay for this group.
        params (list[nn.Parameter]): Parameters belonging to this group.
    """

    name: str
    lr: float
    weight_decay: float = 0.0
    params: list[nn.Parameter] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Convert to the dict format ``torch.optim`` constructors accept."""
        return {
            "name": self.name,
            "params": self.params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }


class OptimizerBuilder:
    """Builds an optimizer with separate param-groups per task head.

    Parameters:
        base_lr (float): Default learning rate (backbone + tasks without override).
        base_weight_decay (float): Default weight decay.
        optimizer_cls (type[torch.optim.Optimizer]): Optimizer class (default: AdamW).
        extra_kwargs (dict[str, Any] | None): Extra constructor args passed to every
            param-group (e.g. ``momentum`` for SGD, ``betas`` for Adam).
    """

    def __init__(
        self,
        base_lr: float,
        base_weight_decay: float = 0.0,
        optimizer_cls: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._base_lr = base_lr
        self._base_weight_decay = base_weight_decay
        self._optimizer_cls = optimizer_cls
        self._extra_kwargs = extra_kwargs or {}

    @classmethod
    def from_name(
        cls,
        name: str,
        base_lr: float,
        base_weight_decay: float = 0.0,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> OptimizerBuilder:
        """Build from an optimizer registry key (e.g. ``"adamw"``/``"sgd"``).

        Parameters:
            name (str): Key in the ``optimizers`` registry.
            base_lr (float): Default learning rate.
            base_weight_decay (float): Default weight decay.
            extra_kwargs (dict[str, Any] | None): Extra constructor args (momentum, ...).

        Returns:
            OptimizerBuilder: Builder bound to the resolved optimizer class.
        """
        return cls(
            base_lr=base_lr,
            base_weight_decay=base_weight_decay,
            optimizer_cls=optimizers.get(name),
            extra_kwargs=extra_kwargs,
        )

    def build(
        self,
        model: nn.Module,
        task_lr_overrides: dict[str, float] | None = None,
        task_wd_overrides: dict[str, float] | None = None,
    ) -> torch.optim.Optimizer:
        """Construct an optimizer with per-head param-groups.

        The backbone (everything not inside ``model.heads``) always forms a
        single group at ``base_lr``. Each head with an LR override gets its own
        group; heads without overrides are folded into the backbone group.

        Parameters:
            model (nn.Module): The ``CompositeModel`` (must have a ``heads`` attribute).
            task_lr_overrides (dict[str, float] | None): Per-task LR; task names as keys.
            task_wd_overrides (dict[str, float] | None): Per-task weight decay.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        overrides_lr = task_lr_overrides or {}
        overrides_wd = task_wd_overrides or {}

        heads: nn.ModuleDict | None = getattr(model, "heads", None)
        if heads is None or not overrides_lr:
            return self._optimizer_cls(
                model.parameters(),
                lr=self._base_lr,
                weight_decay=self._base_weight_decay,
                **self._extra_kwargs,
            )

        head_param_ids: set[int] = set()
        task_groups: list[ParamGroupSpec] = []

        for task_name, head_module in heads.items():
            if task_name not in overrides_lr:
                continue
            params = list(head_module.parameters())
            head_param_ids.update(id(p) for p in params)
            task_groups.append(
                ParamGroupSpec(
                    name=f"head/{task_name}",
                    lr=overrides_lr[task_name],
                    weight_decay=overrides_wd.get(task_name, self._base_weight_decay),
                    params=params,
                )
            )

        backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
        groups: list[dict[str, object]] = [
            ParamGroupSpec(
                name="backbone",
                lr=self._base_lr,
                weight_decay=self._base_weight_decay,
                params=backbone_params,
            ).as_dict()
        ]
        groups.extend(spec.as_dict() for spec in task_groups)

        return self._optimizer_cls(groups, **self._extra_kwargs)
