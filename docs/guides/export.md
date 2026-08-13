# Export

A run that declares an `export` section ends by writing a deployment artifact
under its own directory, and by proving that artifact still computes what the
model computes.

```yaml
defaults:
  - export: torchscript     # or: none

export:
  - {name: torchscript}     # every knob a format has is its own constructor argument
```

The section is a list, so a run can write several formats, and each entry is an
ordinary component — the same `{name: ..., ...}` grammar as callbacks, losses and
metrics. A format's options are the exporter's constructor arguments, which is
why there is no per-format config schema to keep in step.

## What gets written

One graph: the run's model, taking the inputs declared under `data.inputs` in
declaration order, returning one tensor per task in `tasks` order. A run with a
single task returns that tensor itself rather than a one-element tuple — the
convention every torch model already follows — and the artifact says so in its
signature (`-> Tensor` against `-> ((Tensor, Tensor))`), so a consumer never has
to unpack something that cannot vary. Artifacts land at
`{run.directory}/export/model.<suffix>` — the run directory is the single root
for everything a run produces, so export invents no path of its own.

The example fed to the tracer is synthesized from the run's own `image_size` and
`mean`: those are the fields the transform pipeline hands to `Resize` and
`Normalize`, so it is the shape the model receives. Export therefore needs no
dataset and works from a checkpoint on a machine that has none. A model that
refuses that shape fails immediately, naming the shape and the two fields it came
from.

## Verification is not optional

Every written artifact is read back and run beside the model, at the batch size it
was traced with **and** at batch 1 — the shape deployment actually uses. A graph
that baked its batch is caught by measurement rather than by a list of what is
allowed to be exported. A rich table reports absolute and relative drift per
output; anything outside tolerance fails the run.

Tolerances are the format's own, because the tolerance is knowledge of the
format:

```yaml
export:
  - {name: torchscript, atol: 1.0e-4, rtol: 1.0e-3}
```

## Which weights are shipped

The weights the run stopped on. After `fit`, the checkpoint the run kept goes back
into the module, so the numbers `test` reports and the weights the artifact
carries are the same ones. A run without a checkpoint monitor ships what training
ended with.

Export never loads a checkpoint of its own, and neither does `test` — only `fit`
takes one, and only to continue an interrupted run (`run.resume_path`). To export
a checkpoint without training, name it in `run.checkpoint_path` and turn training
off:

```bash
uv run main.py experiment=examples/classification export=onnx run.train=false +run.checkpoint_path=runs/best.ckpt
```

The `+` is Hydra's, not a typo: `config.yaml` does not declare
`run.checkpoint_path`, and adding a key a group file left out is what `+` is for.
Writing it in an experiment file needs no prefix.

## TorchScript

`torch.jit` is deprecated as an authoring API — torch asks for `torch.compile` or
`torch.export` instead — but the `.pt` format is what Triton's `pytorch_libtorch`
backend loads, so the backend uses it deliberately and silences that one notice.
`torch.jit.script` is not offered: it cannot compile a graph whose forward builds
a `Batch`.

Load an artifact the way any TorchScript consumer does:

```python
import torch

model = torch.jit.load("runs/my-project/2026-08-05/12-00-00/export/model.pt")
model.eval()
label = model(torch.randn(1, 3, 224, 224))   # a tuple, when the run has several tasks
```

### Artifacts that cannot leave the trace device

Tracing freezes any tensor computed *inside* `forward` into a constant pinned to
the device it was traced on. Such an artifact is perfect on CPU and is then
refused by the accelerator it was written for — under `.to(device)` and under
`torch.jit.load(..., map_location=device)` alike. Export runs the written file on
every accelerator the machine has and warns, naming the device and the error:

```
model.pt runs on the trace device but mps refused it (TypeError: Cannot convert a
MPS Tensor to float64 ...). Tracing bakes tensors computed inside forward as
constants pinned to the trace device; build the model so they live in a
registered buffer instead — for a timm ViT that is 'dynamic_img_size: false'
with an explicit 'img_size'.
```

A warning rather than a refusal: the artifact is honest on the device it was
traced on, and one accelerator's limits do not predict another's. The usual cause
is a vision transformer built with `dynamic_img_size: true`, which recomputes its
rotary position grid every forward — `configs/model/dpt_dinov3.yaml` shows the
static form that travels.

## ONNX

```yaml
defaults:
  - export: onnx

export:
  - {name: onnx, opset_version: 18}
```

Written through torch's modern `torch.export`-based exporter, with the batch axis
dynamic — so the same file serves one sample and a whole batch. The weights ride
beside the graph in `model.onnx.data`: **both files travel together**, and a
deployment that copies only `model.onnx` has copied a graph without weights. Pass
`external_data: false` for a single file instead.

`opset_version` defaults to 18 because that is the modern exporter's floor. Asking
for less does not fail on its own — torch falls back to 18 and attempts a
down-conversion that may quietly not happen — so the backend reads the opset back
out of the written file and refuses a mismatch. An opset below 18 needs the
deprecated exporter, which this backend does not offer.

Every other knob of `torch.onnx.export` forwards untouched — `optimize`, `report`,
`verbose`, `keep_initializers_as_inputs` and the rest are reachable by name
without this framework declaring them.

### Uniform tensor names

By default the file speaks the run's vocabulary: the inputs are named after
`data.inputs`, the outputs after the tasks. A serving stack that wraps many models
behind one contract wants the opposite — the same names whatever is inside:

```yaml
export:
  - {name: onnx, tensor_names: uniform}
```

One input becomes `input` and one output `output`; several become `input_0`,
`input_1`, `output_0`, `output_1`, in declaration order. The parity report keeps
naming tasks either way — it tells you which task drifted, not what the file calls
it.

This is a parameter of ONNX rather than of the run because only a format that
stores names can honour it: a traced TorchScript file has none at all, so
declaring it there is refused rather than silently ignored.

### Simplifying

```yaml
export:
  - {name: onnx, simplify: true}
```

Runs onnx-simplifier over the written graph. Off by default because the modern
exporter already optimizes; measured on resnet18 it drops four dead initializers
and 28% of the bytes. The weight file is rewritten rather than appended to — ONNX's
external-data writer appends, which would otherwise grow a 44.70 MB artifact to
76.84 MB with half of it unreachable.

## TensorRT

```yaml
defaults:
  - export: tensorrt

export:
  - {name: tensorrt, precision: fp16, min_batch: 1, opt_batch: 4, max_batch: 8}
```

Writes a serialized engine (`model.plan`) for Triton's `tensorrt_plan` backend,
built through TensorRT's own ONNX parser from an intermediate graph that lands in
a temporary directory — a run that asked for an engine did not ask for an ONNX
file beside it.

The profile is a batch range, not a `[N, C, H, W]` triple: the exported graph
already pins channels and spatial size, so a triple would restate what the model
fixed with room to disagree. `precision: fp16` also sets a looser parity
tolerance, because half precision drifts where a traced graph does not; state
`atol`/`rtol` to override it.

> **CUDA only, and built where it will run.** An engine is specific to the GPU,
> the driver and the TensorRT version it was built against — one built elsewhere
> will not load. `tensorrt` is therefore **not** a declared dependency: NVIDIA
> publishes no macOS build, and its PyPI distribution is a stub that fetches a
> CUDA-matched wheel at build time, so it cannot be locked from every machine.
> The backend checks for it and names the fix: `uv pip install tensorrt`.

If a `.plan` is not the goal, the ONNX artifact is enough on its own: Triton's
`onnxruntime` backend runs it under the TensorRT execution provider, and
`trtexec --onnx=model.onnx --saveEngine=model.plan --fp16` builds an engine from
it by hand.

## Adding a format

One class, registered by decorator — no config schema, no union member, no group
file beyond the one that names it:

```python
@exporter_registry.register("onnx")
class OnnxExporter(Exporter):
    def __init__(self, opset_version: int = 17, atol: float = 1e-4, rtol: float = 1e-3) -> None:
        super().__init__(atol=atol, rtol=rtol)
        import onnxruntime  # noqa: F401 — a missing runtime must fail at assembly, not after training
        self.opset_version = opset_version

    def export(self, model: DeployableModel, example: tuple[Tensor, ...], destination: Path) -> Path: ...

    def load(self, path: Path) -> Runnable: ...
```

`load` is abstract on purpose: a format nobody can read back leaves "the export
succeeded" unprovable. Importing the third-party runtime in `__init__` is what
makes a missing dependency fail while the experiment is being assembled, rather
than an hour into training.
