"""Reading a task's output in a test that is about the composed family.

``Prediction.outputs`` and ``Batch.targets`` carry whatever shape a task has — a tensor
for a per-sample or per-pixel task, an ``Instances`` for a per-instance one. A test
exercising the composed family knows which of the two it produced, and says so here
rather than at sixty call sites, so the narrowing reads as the claim it is instead of a
cast bolted onto an assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from src.core import Instances

if TYPE_CHECKING:
    from src.core import TaskOutput


def tensor(value: TaskOutput | None) -> Tensor:
    """The task output as the tensor the composed family produced.

    Fails the test rather than raising a framework error: reaching this with anything
    else means the fixture built a family the test was not written for, which is a
    mistake in the test and should read like one.
    """
    assert isinstance(value, Tensor), f"expected a tensor task output, got {type(value).__name__}"
    return value


def instances(value: TaskOutput | None) -> Instances:
    """The task output as the set of objects a per-instance task produced.

    The other half of the same claim ``tensor`` makes, and it reads the same way: a test
    about detection says so once, here, rather than narrowing at every assertion.
    """
    assert isinstance(value, Instances), f"expected an Instances task output, got {type(value).__name__}"
    return value
