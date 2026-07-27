# Core concepts

## Core concepts

**A Task is a composition of three orthogonal axes.** A *topology* defines the output
structure (global per-sample, dense per-pixel, ranking / multistream for embeddings); an
*objective* defines label semantics (multiclass / multilabel / binary / continuous / metric);
a *modality* defines the input side (image / precomputed embedding / multi-encoder). Familiar
names — `classification`, `segmentation`, `regression`, `triplet`, `contrastive` — are thin
presets over this composition. `segmentation(objective="multilabel")` works out of the box with
no extra code; adding a new variant is one `objective:` change in YAML.

**`num_classes` is never hardcoded.** The data module reads and fits target encoders at
setup time, populates a `RuntimeContext`, and only then are tasks and model heads built
with concrete output dimensions. Class counts flow from data → runtime → model
automatically.

**Hydra groups = swappable building blocks.** Backbone, optimizer, scheduler, dataloader,
transforms, logger, callbacks, trainer, and export are independent config groups. Combine
them freely; override any key via CLI without touching shared config files.

**Train → test → export, one pipeline.** A run executes `fit`, `test`, and model `export`
(ONNX / TorchScript / TensorRT with numerical-parity verification), each gated by a `run_*` flag.
`SampleLogCallback` renders ground-truth-vs-prediction grids to interactive HTML along the way.

**Training regimes compose.** Turning on [knowledge distillation](guides/distillation.md)
(`distillation:` — per-task loss becomes `hard + weight · KL` against frozen teachers) or
[LoRA fine-tuning](guides/lora.md) (`lora:` — frozen backbone + low-rank adapters, merged
back into plain weights at export) is one YAML section each; both compose with EMA, freeze,
batch transforms, and loss-parameter scheduling without special setup.

---

## How components are built


Most of the config maps directly onto Python objects. There are **two construction
families** — knowing which one a section uses tells you how to customize it.

**1. Typed sections** — a fixed schema with one dedicated builder. A `kind` (or `name`)
field selects the registry adapter; the remaining fields are forwarded to it as
constructor arguments. Used by `backbone`, `optimizer`, `scheduler`, `data`, `dataloader`,
`logger`, `trainer`.

```yaml
backbone: {kind: smp, name: unet, encoder_name: resnet34}   # kind → adapter; encoder_name forwarded
optimizer: {name: adamw, lr: ${lr}, weight_decay: 1.0e-4}    # name → optimizer class; rest forwarded
```

Typed sections are `extra="allow"`: unknown keys forward verbatim to the underlying
constructor (smp's `encoder_name`, an optimizer's `momentum`, a DataLoader's `timeout`).

**2. Brick-specs** — free-form, with three interchangeable forms. Used by `loss`,
`metrics`, `target_encoder`, `head`, `callbacks`, the `transform` inside a batch
transform, and `trainer.profiler`.

| Form | YAML | Meaning |
|---|---|---|
| string | `loss: cross_entropy` | registry key, default args |
| name + params | `loss: {name: cross_entropy, label_smoothing: 0.1}` | registry key + kwargs |
| `_target_` | `loss: {_target_: my_pkg.MyLoss, alpha: 0.3}` | import path, no registration needed |

The first two forms look the component up in a **registry** (short, discoverable names);
`_target_` imports any class by dotted path — the escape hatch for code you didn't
register. Both reach the same constructor; pick by whether the thing is registered.

**Nested graphs.** A `_target_` spec is resolved recursively, so object trees can be
built inline (e.g. an Albumentations pipeline):

```yaml
transforms:
  train:
    _target_: albumentations.Compose
    transforms:
      - {_target_: albumentations.HorizontalFlip}
      - {_target_: albumentations.Normalize}
      - {_target_: albumentations.pytorch.ToTensorV2}
```

Inside a `_target_`, only `_target_` is available — registry short-names are a
top-level convenience.

**`trainer.profiler` mixes both.** `trainer` is a typed section, but its `profiler`
sub-key is a brick-spec: a string alias (`profiler: simple`) passes straight to
Lightning, while a `_target_` mapping is instantiated so the profiler can declare its
own output path:

```yaml
trainer:
  profiler:
    _target_: lightning.pytorch.profilers.AdvancedProfiler
    dirpath: ${save_dir}     # write the report under the run directory
    filename: profile
```

**Runtime values are injected, never written.** `num_classes` and similar are inferred
from data at `setup()` and injected into the components that need them — which is why you
never write `num_classes` in a loss / metric / transform spec. Any param you set
explicitly overrides an injected default.

**To customize a component** (both shown in [Extending the framework](reference/extending.md)):
register your class under a short key (`@registry.register("my_key")`) and use the `name`
form, **or** skip registration and point `_target_` straight at it.

> Unlike raw Hydra, `_partial_` and positional `_args_` are not supported — components
> take keyword arguments.
