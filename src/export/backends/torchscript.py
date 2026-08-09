"""TorchScript export: the one format torch itself can write, load and prove."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.export.exporters import Exporter
from src.export.registry import exporter_registry

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from src.export.deployable import DeployableModel

log = logging.getLogger(__name__)


@contextmanager
def _without_the_deprecation_notice() -> Iterator[None]:
    """Silence torch's ``torch.jit`` deprecation notice, and only that one.

    Scoped to our own calls so a deprecation raised by anything else still
    reaches the user, who can act on those.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"`torch\.jit\.", category=DeprecationWarning)
        yield


@dataclass(frozen=True, slots=True)
class _CompiledGraph:
    """Runs a written TorchScript file positionally, as the ``Runnable`` contract asks."""

    module: torch.jit.ScriptModule

    def __call__(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        with torch.no_grad(), _without_the_deprecation_notice():
            written = self.module(*inputs)
        return written if isinstance(written, tuple) else (written,)


@exporter_registry.register("torchscript")
class TorchScriptExporter(Exporter):
    """Traces the graph with its example inputs and saves it as ``.pt``.

    ``torch.jit`` is deprecated as an authoring API — measured on torch 2.13, ``trace``,
    ``script``, ``save`` and ``load`` each emit a ``DeprecationWarning`` pointing at
    ``torch.compile`` / ``torch.export``. The *format* is not deprecated: Triton's
    ``pytorch_libtorch`` backend loads ``.pt``, which is why this exists and why the
    notice is silenced around our own calls rather than handed to a user who cannot act
    on it.

    Tracing only. Measured, ``torch.jit.script`` cannot compile a graph whose forward
    builds a ``Batch`` — it raises ``OSError: Failed to get source for ...`` — and a
    knob whose only outcome is a politer failure is not a knob.
    """

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        # Not with_suffix: a destination whose name carries a dot ('model.v2') would lose it.
        path = destination.parent / f"{destination.name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad(), _without_the_deprecation_notice():
            compiled = torch.jit.trace(model, example)
            torch.jit.save(compiled, str(path))
        _report_where_it_travels(path, example)
        return path

    def load(self, path: Path) -> _CompiledGraph:
        with _without_the_deprecation_notice():
            module = torch.jit.load(str(path))
        module.eval()
        return _CompiledGraph(module)


def accelerators() -> list[str]:
    """The devices other than the trace device this machine could deploy to."""
    available = []
    if torch.cuda.is_available():
        available.append("cuda")
    if torch.backends.mps.is_available():
        available.append("mps")
    return available


def _report_where_it_travels(path: Path, example: tuple[Tensor, ...]) -> None:
    """Run the written artifact on every accelerator this machine has, and say what refused it.

    ``torch.jit.trace`` bakes any tensor computed inside ``forward`` as a constant
    pinned to the trace device, so an artifact that is perfect on CPU can fail the
    moment it reaches the accelerator it was written for. Measured on a timm ViT
    with rotary embeddings and ``dynamic_img_size=True``: correct on CPU, refused
    after ``.to()``, and refused again under ``map_location`` — the workaround
    usually recommended for it. Built statically (``dynamic_img_size: false`` with
    an explicit ``img_size``) the same model travels.

    Measured rather than recognised: sniffing a model for the attributes that
    cause it would name one cause of a general problem and would tie this file to
    another library's internals.

    Said rather than raised: the artifact is honest on the device it was traced
    on, and one accelerator's limits do not predict another's — the same file was
    refused by MPS over a float64 constant, a dtype CUDA supports.
    """
    for device in accelerators():
        try:
            with _without_the_deprecation_notice():
                moved = torch.jit.load(str(path), map_location=device)
            moved.eval()
            with torch.no_grad():
                moved(*(tensor.to(device) for tensor in example))
        except Exception as error:  # noqa: BLE001 — any refusal is the report, whatever its shape
            log.warning(
                "%s runs on the trace device but %s refused it (%s: %s). Tracing bakes tensors computed "
                "inside forward as constants pinned to the trace device; build the model so they live in "
                "a registered buffer instead — for a timm ViT that is 'dynamic_img_size: false' with an "
                "explicit 'img_size'.",
                path.name,
                device,
                type(error).__name__,
                str(error).splitlines()[0][:160],
            )
