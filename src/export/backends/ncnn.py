"""ncnn: Tencent's mobile runtime — a text graph beside its weights, converted from TorchScript by pnnx."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import numpy
import torch
from torch import Tensor

from src.export.backends.torchscript import TorchScriptExporter
from src.export.base import ABSOLUTE_TOLERANCE, BATCH_AXIS, RELATIVE_TOLERANCE, Exporter, Runnable
from src.export.deployable import DeployableModel
from src.export.registry import exporter_registry

WEIGHTS = "bin"
"""What an ncnn graph's weights are called, sitting beside the graph itself.

One home, because three readers need it: the export that writes the pair, the reader that opens both,
and the record that has to tell a deployment this artifact does not travel alone.
"""

DEPLOYMENT_BATCH = 1
"""The second shape the converter is given, so it does not settle the batch into the graph.

pnnx reads what varies between two shapes; one row against the example's own is the difference that
matters, and it is the size a phone actually sends.
"""


def require_pnnx() -> Path:
    """The converter, or a refusal naming it — the binary that turns TorchScript into an ncnn graph.

    Not a dependency of this framework: it ships a platform-specific binary, and the framework cannot
    hold a build for every platform it runs on. Everything a declaration can get wrong is settled before
    this is reached, so a bad one answers on any machine.
    """
    try:
        import pnnx
    except ImportError as error:
        raise ImportError(
            "pnnx is not installed, so this run cannot convert anything to ncnn. It is `pip install "
            "pnnx`, which ships the converter binary, and it is not a dependency of this framework "
            "because that binary is built per platform."
        ) from error
    return Path(pnnx.EXEC_PATH)


@exporter_registry.register("ncnn")
class NcnnExporter(Exporter):
    """A graph for phones: ``.param`` says what the network is, ``.bin`` holds its weights.

    Converted by pnnx from the TorchScript its own declaration describes, which is a *declared* step for
    the reason the engine gives: an exporter constructed privately would drop whatever a run said about
    it. It is written to a scratch directory, so a run that also declares ``torchscript`` keeps the
    artifact its own declaration asked for.

    The names do not survive: ncnn blobs are pnnx's (``in0``, ``out0``), not this run's task names, so a
    deployment reads them out of the ``.param`` and a record of this artifact has to say the order.

    Nothing below ``require_pnnx`` has been measured. The converter ships as a platform binary, and the
    one in this environment refuses to start (built for macOS 15, and this is 14.3) — so the conversion
    is written from pnnx's documented interface and proven by nobody. The reader below is unmeasured for
    the same reason: with no artifact to open, a hand-written ncnn graph was tried and crashed the
    process outright, which is why the return codes are checked before anything else touches the net.
    """

    suffix: ClassVar[str] = "param"

    def __init__(
        self,
        *,
        fp16: bool = False,
        torchscript: TorchScriptExporter | None = None,
        atol: float = ABSOLUTE_TOLERANCE,
        rtol: float = RELATIVE_TOLERANCE,
    ) -> None:
        super().__init__(atol=atol, rtol=rtol)
        self.fp16 = fp16
        if torchscript is not None and not isinstance(torchscript, TorchScriptExporter):
            raise TypeError(
                f"`torchscript` was declared as {type(torchscript).__name__}, and pnnx converts a traced "
                "TorchScript graph and no other format. Declare the intermediate step as a "
                "TorchScriptExporter, or leave it out and this builds its own."
            )
        self.torchscript = torchscript if torchscript is not None else TorchScriptExporter()

    def weights_of(self, path: Path) -> Path:
        """Where this graph's weights sit: beside it, under the same name, and it is useless without them."""
        return path.with_suffix(f".{WEIGHTS}")

    def travels_with(self, path: Path) -> tuple[Path, ...]:
        """Always the weights: the graph says what the network is, and these are what it knows."""
        return (self.weights_of(path),)

    def describe(self, path: Path) -> Mapping[str, object]:
        """Half precision is what a phone is given an ncnn graph for, so a deployment is told which it got.

        The blob names are not recorded: they are pnnx's (``in0``, ``out0``) rather than this run's, and
        the record already says what each position means, in the order every format answers in.
        """
        return {"fp16": self.fp16}

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        converter = require_pnnx()
        with TemporaryDirectory() as scratch:
            traced = self.torchscript.export(graph, example, Path(scratch) / "graph")
            self._convert(converter, traced, example, path, Path(scratch))

    def _convert(self, converter: Path, traced: Path, example: tuple[Tensor, ...], path: Path, scratch: Path) -> None:
        """Run the converter over the traced graph, at two shapes, and refuse whatever it did not write.

        Run as the tool rather than through ``pnnx.convert``: that wrapper does not look at what the
        binary returned and then imports the Python file the conversion was supposed to generate, so a
        converter that could not start at all reports a missing file and says nothing about why —
        measured here, where it cannot start. It also executes generated code, which an export has no
        reason to do. Everything else pnnx would write lands in the scratch directory it runs in.
        """
        said = subprocess.run(
            [
                str(converter),
                traced.name,
                f"inputshape={_shapes(example, int(example[0].shape[BATCH_AXIS]))}",
                f"inputshape2={_shapes(example, DEPLOYMENT_BATCH)}",
                # Absolute, because the converter runs somewhere else: a destination as the caller wrote
                # it — and the shipped run directory is relative — would resolve against the scratch
                # directory below and be swept away with it, after an exit code saying all was well.
                f"ncnnparam={path.resolve()}",
                f"ncnnbin={self.weights_of(path).resolve()}",
                f"fp16={int(self.fp16)}",
            ],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
        missing = [one.name for one in (path, self.weights_of(path)) if not one.exists()]
        if said.returncode != 0 or missing:
            trouble = (said.stderr or said.stdout or "").strip().splitlines()
            raise RuntimeError(
                f"pnnx wrote no ncnn graph for {traced.name} (exit {said.returncode}"
                f"{', missing ' + ', '.join(missing) if missing else ''}). It said: "
                f"{' | '.join(trouble[-3:]) or 'nothing at all'}"
            )

    def load(self, path: Path) -> Runnable:
        import ncnn

        net = ncnn.Net()
        weights = self.weights_of(path)
        if net.load_param(str(path)) != 0 or net.load_model(str(weights)) != 0:
            raise ValueError(
                f"ncnn would not read {path.name} with {weights.name}: a graph and its weights travel "
                "together, and what it made of them is on its own error stream above. Measured: going on "
                "from here crashes the process rather than answering."
            )
        fed, answers = net.input_names(), net.output_names()

        def run(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            """One sample at a time, because an ncnn ``Mat`` has no batch axis — the rows are this loop."""
            rows = []
            for index in range(int(tensors[0].shape[BATCH_AXIS])):
                with net.create_extractor() as extractor:
                    for name, tensor in zip(fed, tensors, strict=True):
                        extractor.input(name, ncnn.Mat(tensor[index].detach().cpu().numpy().copy()))
                    rows.append([_extracted(extractor, name, path) for name in answers])
            return tuple(torch.from_numpy(numpy.stack([row[at] for row in rows])) for at in range(len(answers)))

        return run


def _extracted(extractor: Any, name: str, path: Path) -> numpy.ndarray:
    """One output blob, or a refusal — ncnn answers a code beside the value and the code is the truth.

    Checked for the same reason the two loads above are: a failed extraction hands back an empty ``Mat``,
    which stacks into a shape nothing else explains, and the comparison that follows would be measuring
    the absence rather than the artifact. Unmeasured like everything else below ``require_pnnx``.
    """
    said, blob = extractor.extract(name)
    if said != 0:
        raise RuntimeError(
            f"ncnn answered {said} for output {name!r} of {path.name} rather than a value, so there is "
            "nothing here to compare with the model."
        )
    return numpy.array(blob)


def _shapes(example: tuple[Tensor, ...], batch: int) -> str:
    """The converter's spelling of a batch of inputs: ``[2,3,224,224],[2,4]``, one bracket per input."""
    return ",".join(
        "[" + ",".join(str(size) for size in (batch, *one.shape[BATCH_AXIS + 1 :])) + "]" for one in example
    )
