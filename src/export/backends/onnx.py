"""The portable format: one graph, named tensors, a symbolic batch, read by any ONNX runtime."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor

from src.export.base import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, Exporter, Runnable, batch_may_vary
from src.export.deployable import DeployableModel
from src.export.registry import exporter_registry

STANDARD_DOMAIN = ("", "ai.onnx")
"""How the base operator set names itself in a file: unset, or spelled out."""

WEIGHTS_BESIDE_IT = "data"
"""What torch calls the file it writes the weights into when they do not fit inside the graph.

One home for three readers: the export that decides whether there is such a file, the record that has to
tell a deployment this artifact travels as a pair, and anyone copying one of them somewhere else.
"""


CONVERTERS = ("onnxscript", "onnx_ir")
"""The libraries torch converts a graph through, which narrate what they do as they do it."""


@contextmanager
def _without_the_converters_own_notes() -> Iterator[None]:
    """Keep the graph converters' INFO notes out of a run's log, around this module's own calls only.

    Measured on torch 2.13 with onnxscript 0.7.1: writing one artifact makes six records — nodes folded,
    unused nodes removed, rewrite rules applied — and they arrive over the top of what the run produced.
    ``verbose=False`` does not touch them; measured, the count is the same either way. Anything at
    warning or above still passes, which is the half a reader can act on.
    """
    quieted = [logging.getLogger(name) for name in CONVERTERS]
    levels = [one.level for one in quieted]
    for one in quieted:
        one.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for one, level in zip(quieted, levels, strict=True):
            one.setLevel(level)


@exporter_registry.register("onnx")
class OnnxExporter(Exporter):
    """Written through ``torch.onnx.export``, which since torch 2.9 captures the graph with ``torch.export``.

    That is the same front end :class:`Pt2Exporter` uses, so both formats are the same graph: the
    names this run declared land in the file, and the batch axis is symbolic there too.

    Parameters:
        opset: The operator set a deployment runtime needs. Left unset, torch picks its own current one,
            which is the right answer whenever nothing downstream insists otherwise.
        external_data: Whether weights too large to sit comfortably in the graph are written next to
            it instead. Off, so an artifact is one file that travels by itself; a model above the 2 GiB
            a protobuf can hold has no choice and says so here.
        simplify: Whether to fold what the graph computes from constants, after torch's own optimizer.
            Off, because on a *trained* model it buys almost nothing — see the measurement below — and a
            rewrite by another library is not something to do to every artifact for 0.3%.
    """

    suffix: ClassVar[str] = "onnx"

    def __init__(
        self,
        *,
        opset: int | None = None,
        external_data: bool = False,
        simplify: bool = False,
        atol: float = ABSOLUTE_TOLERANCE,
        rtol: float = RELATIVE_TOLERANCE,
        verify: bool = True,
    ) -> None:
        super().__init__(atol=atol, rtol=rtol, verify=verify)
        self.opset = opset
        self.external_data = external_data
        self.simplify = simplify

    def write(self, graph: DeployableModel, example: tuple[Tensor, ...], path: Path) -> None:
        with torch.no_grad(), _without_the_converters_own_notes():
            torch.onnx.export(
                graph,
                example,
                str(path),
                input_names=list(graph.input_names),
                output_names=list(graph.output_names),
                dynamic_shapes=batch_may_vary(example),
                opset_version=self.opset,
                external_data=self.external_data,
                # The default since torch 2.9, and everything above depends on it: the older exporter
                # traced, named its tensors by position, and took `dynamic_axes` instead.
                dynamo=True,
                # Not a knob: what it turns off is torch's own progress trace, and what a run has to
                # say about an export is the record it writes and the table drawn from it.
                verbose=False,
            )
        if self.simplify:
            self._fold_what_is_constant(path)
        self._refuse_an_opset_the_file_does_not_carry(path)

    def _fold_what_is_constant(self, path: Path) -> None:
        """Rewrite the artifact through onnx-simplifier, leaving one coherent set of files.

        Measured on onnxsim 0.7.0, over graphs torch's own optimizer had already been through, and the
        measurement is worth stating carefully because the obvious one misleads. The same resnet18 loses
        **28.3%** of its bytes untrained and **0.3%** trained: at initialization batch norm is the
        identity — unit variance, zero mean — so folding it into the convolution before it collapses
        whole tensors, and once the run has moved those numbers there is nothing left to collapse. The
        reference implementation recorded the first figure, which is not the one a shipped model gets.

        What remains is worth having where a graph really does compute something from constants, and it
        is proven against the model afterwards like every other artifact — so this is a knob, and off.

        The sidecar is removed first: ONNX's external-data writer *appends* to a location file that
        already exists, and writing over a live one grew a 44.70 MB artifact to 76.84 MB, half of it
        unreachable.
        """
        import onnx
        from onnxsim import simplify

        folded, agreed = simplify(onnx.load(str(path)))
        if not agreed:
            raise RuntimeError(
                f"onnxsim could not confirm that its own rewrite of {path.name} answers as the graph it "
                "was given. What torch wrote is the model; declare `simplify: false` to keep that one."
            )
        sidecar = _weights_beside(path)
        apart = sidecar.exists()
        sidecar.unlink(missing_ok=True)
        if apart:
            onnx.save(folded, str(path), save_as_external_data=True, location=sidecar.name)
        else:
            onnx.save(folded, str(path))

    def travels_with(self, path: Path) -> tuple[Path, ...]:
        """Nothing, unless the weights were written beside the graph — then neither half is the model.

        Read off the disk rather than off the declaration: asking for the weights apart is asking for
        what does not fit comfortably inside the graph, and a model with nothing that large keeps them
        in. A record that promised a file nobody wrote would send a deployment looking for it.
        """
        sidecar = _weights_beside(path)
        return (sidecar,) if sidecar.exists() else ()

    def describe(self, path: Path) -> Mapping[str, object]:
        """The operator set the file carries, which is what a runtime has to support to load it.

        Not whether the weights sit beside it: ``travels_with`` above already answers that, and saying it
        twice is two statements free to disagree.
        """
        return {"opset": _opset_of(path), "simplified": self.simplify}

    def _refuse_an_opset_the_file_does_not_carry(self, path: Path) -> None:
        """Read back what was written, because torch substitutes an opset it cannot reach and says nothing.

        Measured on torch 2.13 with onnx 1.22: ``opset_version=9`` and ``opset_version=99`` both wrote a
        file carrying 18, with the version converter's complaint logged and the export reported as
        successful. A run names an opset because some runtime needs that one, so a file shipped under a
        version it does not have is the failure this exists to stop — and checked here rather than before
        writing, because which opsets a model can reach depends on the operators in it.
        """
        if self.opset is None:
            return
        carried = _opset_of(path)
        if self.opset != carried:
            raise ValueError(
                f"{path.name} was asked for opset {self.opset} and carries {carried} instead: torch "
                "replaces an opset it cannot convert this model to, silently. Declare one this model can "
                "be written at. The file is written; it is the run that cannot claim it is the version "
                "asked for."
            )

    def load(self, path: Path) -> Runnable:
        import onnxruntime

        session = onnxruntime.InferenceSession(str(path))
        # Read off the file rather than off the graph, because a runtime is fed by the names the artifact
        # carries and those are the only ones it will answer to. Paired with the caller's tensors in order,
        # so what this settles is their count; that they are the names this run declared is a separate
        # claim, and the test that opens the file holds it.
        fed = [one.name for one in session.get_inputs()]

        def run(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
            given = {name: one.detach().cpu().numpy() for name, one in zip(fed, tensors, strict=True)}
            return tuple(torch.from_numpy(one) for one in session.run(None, given))

        return run


def _weights_beside(path: Path) -> Path:
    """Where torch puts an ONNX model's weights when it is told to keep them out of the graph."""
    return Path(f"{path}.{WEIGHTS_BESIDE_IT}")


def _opset_of(path: Path) -> int:
    """Which operator set a written file actually carries, whatever the declaration asked for."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    carried = {one.version for one in model.opset_import if one.domain in STANDARD_DOMAIN}
    return max(carried, default=0)
