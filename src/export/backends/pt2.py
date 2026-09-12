"""torch's own deployment format: the graph captured as a program and saved as a PT2 archive."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from src.export.base import Exporter, Runnable, batch_may_vary
from src.export.deployable import DeployableModel, as_outputs
from src.export.registry import exporter_registry


@exporter_registry.register("pt2")
class Pt2Exporter(Exporter):
    """``torch.export`` captures the graph; ``torch.export.save`` writes it; ``torch.export.load`` reads it.

    Not TorchScript, and the reason is measured rather than stylistic. ``torch.jit.script`` cannot compile
    this graph at all — the three refusals are named in :class:`DeployableModel` — and ``torch.jit.trace``,
    which the reference implementation shipped, is deprecated as of torch 2.13: it still writes a working
    file and warns that it will not always.

    What this route adds is not a workaround for that. A traced graph can bake in the batch it was traced
    at while saying nothing about it; an exported program carries the batch as a symbol, so the file
    itself promises to serve one row. Measured: saved here, loaded in a fresh interpreter, and run at a
    batch size it was never written at.

    The cost, stated because someone will meet it: a PT2 archive is read by ``torch.export.load``, which
    needs Python and torch. TorchScript could be loaded by libtorch from C++, and this cannot.
    """

    suffix: ClassVar[str] = "pt2"

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        with torch.no_grad():
            program = torch.export.export(graph, example, dynamic_shapes=batch_may_vary(example))
        torch.export.save(program, path)

    def load(self, path: Path) -> Runnable:
        module = torch.export.load(path).module()

        def run(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            with torch.no_grad():
                return as_outputs(cast(Any, module(*tensors)))

        return run
