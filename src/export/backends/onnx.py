"""ONNX export through torch's modern, ``torch.export``-based exporter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from src.core.choices import one_of
from src.export.exporters import Exporter
from src.export.registry import exporter_registry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from torch import Tensor

    from src.export.deployable import DeployableModel

type TensorNames = Literal["declared", "uniform"]
"""What a written graph calls its tensors.

``declared`` keeps the run's own vocabulary — 'image' goes in, 'label' comes out.
``uniform`` gives every model one interface ('input'/'output' alone, 'input_0',
'input_1' … in company), which is what a serving wrapper needs when it must
accept any model without knowing which task is inside; the price is that a
deployment config no longer says 'label'.
"""

BATCH_AXIS = "batch"
"""What the dynamic leading dimension is called in the written graph."""


@dataclass(frozen=True, slots=True)
class _OnnxSession:
    """Runs a written ONNX file, feeding by the names the file itself declares.

    Reading the names back out of the artifact rather than taking them from the
    caller is what proves they landed in the graph.
    """

    session: Any  # onnxruntime.InferenceSession — onnxruntime ships no type stubs, so Any is honest

    def __call__(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        feeds = {
            declared.name: tensor.detach().cpu().numpy()
            for declared, tensor in zip(self.session.get_inputs(), inputs, strict=True)
        }
        return tuple(torch.from_numpy(written) for written in self.session.run(None, feeds))


@exporter_registry.register("onnx")
class OnnxExporter(Exporter):
    """Writes an ONNX graph with a dynamic batch axis, and reads it back with onnxruntime.

    Exactly one of ``torch.onnx.export``'s twenty-five parameters cannot be written by
    hand: ``dynamic_shapes`` depends on how many inputs the graph has *and* on the
    nesting ``torch.export`` imposes on a ``*inputs`` forward — measured, a flat spec
    raises "inputs[0] is a <class 'tuple'>, but dynamic_shapes[0] is a <class 'dict'>".
    Converting that is this wrapper's whole job; everything else forwards verbatim, so
    ``external_data``, ``optimize``, ``report`` and the rest stay reachable.

    The legacy TorchScript-based exporter is refused. It is the only way to an opset
    below 18, but torch says it "will be removed" and it cannot read the dynamic shapes
    built here — supporting both would mean two dynamic-axis vocabularies for a path
    with an expiry date.

    Parameters:
        opset_version (int): Operator set the graph is written at. 18 rather than
            something older because 18 is the modern exporter's floor — measured,
            asking for 17 lands 18 after a failed down-conversion, so the written
            file is checked against this number and a mismatch is refused.
        tensor_names (str): ``declared`` keeps the run's vocabulary; ``uniform``
            gives every model one interface. Lives here rather than on the run
            because only a format that stores names can honour it — a traced
            TorchScript file has none.
        simplify (bool): Run onnx-simplifier over the written graph. Off by
            default: the modern exporter already runs its own optimizer, and this
            is a second opinion rather than a duty. Measured on resnet18, it drops
            four dead initializers and 28% of the bytes.
        atol (float): Absolute output error tolerated against the source model.
        rtol (float): Relative output error tolerated against the source model.
        **kwargs: Forwarded verbatim to ``torch.onnx.export``.
    """

    def __init__(
        self,
        opset_version: int = 18,
        tensor_names: TensorNames = "declared",
        simplify: bool = False,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(atol=atol, rtol=rtol)
        self.tensor_names = one_of(tensor_names, TensorNames)
        if kwargs.get("dynamo") is False:
            raise ValueError(
                "The legacy ONNX exporter cannot read the dynamic shapes this backend builds, and "
                "torch is removing it. Drop 'dynamo: false'; for an opset below 18 the export has "
                "to be reconsidered rather than downgraded."
            )
        # Imported here, not at module scope: this package sits on the import path of every run,
        # the ONNX stack costs 460 ms (measured), and a stale environment must still fail while
        # the experiment is assembled rather than an hour into training.
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        import onnxscript  # noqa: F401

        self.opset_version = opset_version
        self.simplify = simplify
        self._options = kwargs

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path:
        path = destination.parent / f"{destination.name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        batch = torch.export.Dim(BATCH_AXIS)
        with torch.no_grad():
            torch.onnx.export(
                model,
                example,
                str(path),
                input_names=self._written_names(model.input_names, "input"),
                output_names=self._written_names(model.output_names, "output"),
                opset_version=self.opset_version,
                # torch.export reads a *inputs forward as one tuple parameter, so the per-tensor
                # specs nest one level deeper than the arguments they describe.
                dynamic_shapes=(tuple({0: batch} for _ in example),),
                **self._options,
            )
        if self.simplify:
            _simplify_in_place(path)
        _prove_the_opset(path, self.opset_version)
        return path

    def load(self, path: Path) -> _OnnxSession:
        import onnxruntime

        return _OnnxSession(onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"]))

    def _written_names(self, declared: Sequence[str], role: str) -> list[str]:
        """What the file calls these tensors, under the scheme this exporter was given."""
        if self.tensor_names == "declared":
            return list(declared)
        return [role] if len(declared) == 1 else [f"{role}_{index}" for index in range(len(declared))]


def _prove_the_opset(path: Path, requested: int) -> None:
    """Refuse an artifact written at an operator set nobody asked for.

    Measured: below 18 the exporter falls back to 18 and attempts a down-conversion
    that raises on a real model — torch logs it and the run carries on, leaving a
    file that answers to a different opset than the config named. A serving stack
    pinned to an opset finds that out in production; this finds it out here.
    """
    import onnx

    written = [
        imported.version
        for imported in onnx.load(str(path), load_external_data=False).opset_import
        if imported.domain in ("", "ai.onnx")
    ]
    if requested not in written:
        raise ValueError(
            f"{path.name} was written at opset {written}, not the requested {requested}: the modern "
            f"ONNX exporter implements 18 and above and could not convert down. Set opset_version "
            f"to 18 or higher."
        )


def _simplify_in_place(path: Path) -> None:
    """Rewrite the artifact through onnx-simplifier, leaving one coherent pair of files.

    The sidecar is removed before the rewrite because ONNX's external-data writer
    *appends* to an existing location file — measured, writing over a live one grew
    a 44.70 MB artifact to 76.84 MB of which half was unreachable.
    """
    import onnx
    from onnxsim import simplify

    simplified, agreed = simplify(onnx.load(str(path)))
    if not agreed:
        raise RuntimeError(f"onnx-simplifier could not validate its own output for {path.name}.")
    sidecar = path.parent / f"{path.name}.data"
    external = sidecar.exists()
    sidecar.unlink(missing_ok=True)
    if external:
        onnx.save(simplified, str(path), save_as_external_data=True, location=sidecar.name)
    else:
        onnx.save(simplified, str(path))
