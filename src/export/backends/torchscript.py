"""TorchScript: the format torch's own C++ runtime loads, written by tracing the graph."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from src.export.base import Exporter, Runnable
from src.export.deployable import DeployableModel, as_outputs
from src.export.registry import exporter_registry


@contextmanager
def _without_the_authoring_notice() -> Iterator[None]:
    """Silence torch's ``torch.jit`` deprecation notice around this module's own calls, and only there.

    What torch deprecated is the authoring API, not the artifact: a ``.pt`` file is what libtorch loads,
    and choosing to write one is a decision this module states in full below. Repeating a notice about a
    decision already made, on every call of a run, teaches a reader to skim warnings — and every other
    deprecation, including ones they can act on, still reaches them.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.jit\.", category=DeprecationWarning)
        yield


@exporter_registry.register("torchscript")
class TorchScriptExporter(Exporter):
    """The graph traced with its example and saved as ``.pt``, which loads without Python.

    That is the whole of why this stands beside :class:`Pt2Exporter`: a PT2 archive is read by
    ``torch.export.load`` and needs torch, while a ``.pt`` is read by libtorch from C++ — which is what
    Triton's ``pytorch_libtorch`` backend and every embedded deployment of torch does.

    Traced, not scripted: measured on torch 2.13, ``torch.jit.script`` cannot compile this graph, and the
    three reasons are named in :class:`DeployableModel`. So the batch axis here is not *declared* free
    the way ``torch.export`` declares it — it is generalized out of the example — which makes this the
    one format where verifying a second batch size does real work rather than confirming a promise.

    One hazard this module cannot check, measured on a timm ViT with rotary embeddings: tracing bakes
    tensors computed inside ``forward`` as constants pinned to the device the trace ran on, so an
    artifact that is perfect on CPU can be refused by the accelerator it was written for. A model with
    such tensors should keep them in a registered buffer instead.
    """

    suffix: ClassVar[str] = "pt"

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        with torch.no_grad(), _without_the_authoring_notice():
            torch.jit.save(torch.jit.trace(graph, example), str(path))

    def load(self, path: Path) -> Runnable:
        with _without_the_authoring_notice():
            module = torch.jit.load(str(path))

        def run(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            with torch.no_grad(), _without_the_authoring_notice():
                return as_outputs(cast(Any, module(*tensors)))

        return run
