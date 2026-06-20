"""Small tracing helpers shared by the pipeline and the export backends."""

from __future__ import annotations

from torch import Tensor


def as_output_tuple(raw: Tensor | tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Normalize a module's forward output to a tuple of tensors."""
    return raw if isinstance(raw, tuple) else (raw,)


def trace_args(example_inputs: tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
    """Return the args form torch tracing APIs expect: a bare tensor for one input, else the tuple.

    ``torch.onnx.export`` and ``torch.jit.trace`` treat a tuple as *multiple* positional
    arguments, so a single-input module must be traced with the bare tensor — passing a
    1-tuple would mis-trace it as a one-element arg list.
    """
    return example_inputs if len(example_inputs) > 1 else example_inputs[0]
