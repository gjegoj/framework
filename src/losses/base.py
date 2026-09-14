"""What a loss is here: raw outputs and a target in, a number and the name it reports under out."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cache
from inspect import signature
from typing import Any, ClassVar, cast, get_args

import torch
from torch import Tensor, nn

from src.core import LossOutput, Representation, drop_feature_axis


class Loss(nn.Module, ABC):
    """One objective over raw outputs; activations belong to a task's postprocessing, never here.

    A loss is an ``nn.Module`` because some of them hold parameters (angular margins keep class
    prototypes), and those must be optimized and checkpointed with the run.

    Attributes:
        log_name: What this term is called in a report. It starts as the loss's own name and is replaced
            by the name the declaration used, so a run reads `loss: dice` and finds `dice` in its metrics.
        reads: What the head's numbers have to be for this objective to mean anything. Almost every one
            of them reads a projection, which is why that is the default; an angular margin is added to
            an angle and has nothing to add to anything else. The pair is checked where both are built.
        reads_per_sample: How many answers of one sample this objective compares. Almost every one reads
            the single answer a head makes, which is why that is the default; an objective learning from
            the batch itself reads the pair a sample gave. Read where a head declared over several
            streams meets it, which is the one place both numbers are known.
        reads_soft_targets: Whether a target blended from two samples — what a mixing transform leaves
            behind — is something this can compare. An objective scoring the one class a sample *is*
            cannot, and says so here rather than discovering it on a batch. Declared by the objective
            rather than read off the task: the same task under an ordinary cross-entropy blends
            perfectly well, so nothing about the task settles it.
    """

    reads: Representation = Representation.PROJECTED
    reads_per_sample: int = 1
    reads_soft_targets: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.log_name = snake_case(type(self).__name__)

    @abstractmethod
    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        """The objective, named, as one scalar."""

    def __call__(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        """Typed delegate to ``nn.Module.__call__``, so hooks run and the type survives."""
        return cast(LossOutput, super().__call__(outputs, targets))

    def reported(self, value: Tensor) -> LossOutput:
        """This loss's number under its own name, which is how every leaf loss answers."""
        return LossOutput.reported(self.log_name, value)


class TorchLoss(Loss):
    """A loss torch or a library already implements, declared rather than written.

    A subclass states which module to compute with and, where the library insists on an argument, what
    this framework defaults it to. Everything a run writes reaches that module verbatim — its own
    documentation is the reference — and an argument the library declares as a tensor may be written as
    a list, which is the only form a config file has.

    Attributes:
        module_type: The module this loss computes with.
        defaults: Arguments the library requires and this framework answers for; a run overrides them.
        squeezes_channel: Drop a width-one channel before comparing, for a loss whose target has none.
    """

    module_type: ClassVar[type[nn.Module]]
    defaults: ClassVar[Mapping[str, Any]] = {}
    squeezes_channel: ClassVar[bool] = False

    def __init__(self, **options: Any) -> None:
        super().__init__()
        declared = {**self.defaults, **options}
        wants_tensors = tensor_arguments(type(self))
        self.module = self.module_type(
            **{
                name: torch.as_tensor(value, dtype=torch.float)
                if name in wants_tensors and value is not None
                else value
                for name, value in declared.items()
            }
        )

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        if self.squeezes_channel:
            outputs = aligned(outputs, targets)
        return self.reported(cast(Tensor, self.module(outputs, targets)))


@cache
def tensor_arguments(loss: type[TorchLoss]) -> frozenset[str]:
    """Constructor arguments the library itself declares as tensors — a per-class weight, and its like.

    Read from the signature rather than listed here: the library is the one that knows, and a list of
    ours would drift from it. Anything it declares as a list of numbers (smp's `classes`) stays a list.
    """
    try:
        accepted = signature(loss.module_type, eval_str=True).parameters
    except (TypeError, ValueError, NameError):
        return frozenset()
    return frozenset(
        name
        for name, parameter in accepted.items()
        if parameter.annotation is Tensor or Tensor in get_args(parameter.annotation)
    )


def aligned(outputs: Tensor, targets: Tensor) -> Tensor:
    """The head's output as its target is shaped: one value per position carries a feature axis, a target none."""
    return drop_feature_axis(outputs) if outputs.ndim == targets.ndim + 1 else outputs


def snake_case(name: str) -> str:
    """`BinaryCrossEntropy` reads as `binary_cross_entropy`: a class name as a log key."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class NamedLoss(Loss):
    """A loss the framework did not write, reached by import path: called, and reported under its name.

    The runtime half of what ``TorchLoss`` does declaratively — that one is told which module to build,
    this one is handed one already built — so a `_target_` to any module comparing two tensors is a
    complete declaration, and it reports under its own class name unless a run renames it.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.log_name = snake_case(type(module).__name__.removesuffix("Loss")) or "loss"

    def forward(self, outputs: Tensor, targets: Tensor) -> LossOutput:
        return self.reported(cast(Tensor, self.module(outputs, targets)))
