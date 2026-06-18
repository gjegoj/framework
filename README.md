# Framework

Configuration-driven **multi-task & multimodal** computer-vision training on top of
PyTorch Lightning · Hydra · Pydantic · timm / smp · albumentations · torchmetrics.

> Classification · segmentation · regression · **metric learning** (ranking / dual-encoder),
> with model **export** (ONNX / TorchScript) and interactive **sample visualization** built in.

---

## Table of contents

- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Configuration guide](#configuration-guide)
  - [How components are built](#how-components-are-built)
  - [Data](#data)
  - [DataLoader & cache](#dataloader--cache)
  - [Tasks & presets](#tasks--presets)
  - [Embeddings & metric learning](#embeddings--metric-learning)
  - [Backbone](#backbone)
  - [Optimizer, LR & scheduler](#optimizer-lr--scheduler)
  - [Callbacks](#callbacks)
  - [Logger](#logger)
  - [Export](#export)
  - [Sample visualization](#sample-visualization)
- [Recipes](#recipes)
- [CLI reference](#cli-reference)
- [Extending the framework](#extending-the-framework)
- [Internals](#internals)

> Section bodies are collapsed by default — click a heading's summary to expand it.

---

## Quick start

```bash
uv sync           # install dependencies
make test         # verify everything works
```

The entry point is `main.py`. All configuration lives in `configs/`.
Run with the built-in debug experiment (synthetic data, CPU, 2 epochs):

```bash
uv run python main.py
```

To point at your own data, override the experiment:

```bash
uv run python main.py +experiment=my_exp
```

---

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
(ONNX / TorchScript with numerical-parity verification), each gated by a `run_*` flag.
`SampleLogCallback` renders ground-truth-vs-prediction grids to interactive HTML along the way.

---

## Configuration guide

### How components are built

<details>
<summary>Two construction families: typed sections vs. brick-specs (<code>name</code> / <code>_target_</code>)</summary>

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

**To customize a component** (both shown in [Extending the framework](#extending-the-framework)):
register your class under a short key (`@registry.register("my_key")`) and use the `name`
form, **or** skip registration and point `_target_` straight at it.

> Unlike raw Hydra, `_partial_` and positional `_args_` are not supported — components
> take keyword arguments.

</details>

### Data

<details>
<summary>Split / pre-split modes · multiple inputs · <code>max_samples</code></summary>

**Split mode** — one file, ratios decide the split:

```yaml
data:
  sources: data/annotations.csv
  inputs: image_path          # shorthand: single image column
  split:
    train: 0.8
    val:   0.1
    test:  0.1
```

**Pre-split mode** — separate files per stage:

```yaml
data:
  sources:
    train: data/train.csv
    val:   data/val.csv
  inputs: image_path
```

**Multiple inputs** (multi-view, multimodal):

```yaml
data:
  inputs:
    image:   image_path        # loader auto-detected from extension
    depth:   depth_path        # another image column
    caption: {column: text_col, loader: text}   # explicit loader
```

**Stratified split** — keep class balance across stages:

```yaml
data:
  split: {train: 0.8, val: 0.1, test: 0.1}
  split_stratify: species      # categorical → classification; numeric → quantile-binned
```

**Cap dataset size** for fast iteration:

```yaml
data:
  max_samples: 500       # int → exactly N rows
  max_samples: 0.1       # float → 10% of data
```

</details>

### DataLoader & cache

<details>
<summary>Worker knobs (config group + presets) and the in-RAM image/mask cache</summary>

`dataloader` is its own config group. Override per-run, swap a preset, or add a block in an
experiment:

```bash
uv run python main.py dataloader.num_workers=8 dataloader.pin_memory=true
uv run python main.py dataloader=performance      # GPU preset: 8 workers, pin_memory, prefetch 4
uv run python main.py dataloader=debug            # num_workers=0 (real tracebacks / breakpoints)
```

| Knob | Meaning |
|---|---|
| `num_workers` | loader subprocesses (`0` = main process, debug-friendly) |
| `pin_memory` | page-locked host memory → faster CPU→GPU copies (CUDA only) |
| `persistent_workers` | keep workers alive between epochs (auto-off at `num_workers=0`) |
| `drop_last` | drop the last incomplete **train** batch (val/test never drop) |
| `prefetch_factor` | batches prefetched per worker (auto-off at `num_workers=0`) |

Extra keys forward verbatim to `torch.utils.data.DataLoader` (e.g. `timeout`,
`multiprocessing_context`); framework-owned keys (`batch_size`/`shuffle`/`collate_fn`/…) are
rejected so per-stage conventions hold.

**In-RAM cache** — decode each image/mask once, warmed in the parent before training and
read-only after (so it stays shared across fork workers). Budget = `min(ram_fraction · free
RAM, max_gb)`:

```yaml
data:
  cache:
    ram_fraction: 0.5     # cap at half of available RAM (0 disables)
    max_gb: 8             # absolute cap in GiB
    workers: 8            # threads used to warm the cache
```

> The cache + multi-worker only share memory under **fork** (Linux). On macOS (spawn),
> pick one: cache with `num_workers=0`, or workers with the cache off.

</details>

### Tasks & presets

<details>
<summary>Classification · segmentation · regression · objective / loss / metric overrides · per-head LR</summary>

Tasks are declared as a named dict. The key becomes the task name used in metric logs
(`label/accuracy/val`), loss logs (`loss/val/label`), and per-head LR overrides.

**Classification** (multiclass by default):

```yaml
tasks:
  species:
    preset: classification
    target: species_col
    class_mapping: {0: cat, 1: dog, 2: cow}   # infers num_classes=3
```

**Segmentation**:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    class_mapping: {0: background, 1: defect, 2: edge}   # infers num_classes=3
```

Or with explicit `num_classes` when class names don't matter:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
```

**Regression**:

```yaml
tasks:
  age:
    preset: regression
    target: age
    dim: 1
```

**Objective override** — same preset, different label semantics:

```yaml
tasks:
  tags:
    preset: classification
    objective: multilabel       # sigmoid + BCE instead of softmax + CE
    target: tags_col
    class_mapping: {0: indoor, 1: outdoor, 2: people}
```

Available objectives: `multiclass` · `multilabel` · `binary` · `continuous` · `metric`
(metric learning — see [Embeddings & metric learning](#embeddings--metric-learning)).

**Custom loss** (registry keys: `cross_entropy` · `bce` · `mse` · `l1` · `dice` ·
`weighted_sum` · `arcface` · metric-learning losses):

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    loss:
      name: weighted_sum
      losses: {cross_entropy: 1.0, dice: 2.0}
```

**Custom metrics**:

```yaml
tasks:
  species:
    preset: classification
    target: species_col
    class_mapping: {0: cat, 1: dog, 2: cow}
    metrics:
      accuracy: null
      per_class_f1:
        name: f1
        average: none           # returns [C] vector → logged per class
      confusion_matrix: null
```

**Per-head learning rate** (see [Optimizer, LR & scheduler](#optimizer-lr--scheduler)):

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    optimizer:
      lr: 1.0e-4                # this head gets its own param group
```

</details>

### Embeddings & metric learning

<details>
<summary>Triplet / pairwise ranking (Siamese) and contrastive (dual-encoder, CLIP/SigLIP)</summary>

Metric-learning tasks have no per-sample class label — supervision comes from the
pair/triplet structure or the batch diagonal. The `metric` objective makes the adapter
pass-through and the activation identity; `num_classes` is reinterpreted as the **embedding
dimension** (the projection-head size). The *loss method* is pinned by the preset.

| Preset | Topology | Default loss | Shape of supervision |
|---|---|---|---|
| `triplet` | RANKING | `triplet_margin` | 3 views: anchor / positive / negative |
| `pairwise_ranking` | RANKING | `margin_ranking` | 2 views ranked against each other |
| `contrastive` | MULTISTREAM | `info_nce` | N separate encoders aligned (InfoNCE / SigLIP) |

**RANKING (Siamese)** — N input views go through *one shared backbone* (stacked to
`[B·N, …]`, reshaped to `[B, N, D]`). The view names come from `data.inputs`:

```yaml
data:
  inputs:
    anchor:   anchor_path
    positive: positive_path
    negative: negative_path

tasks:
  embed:
    preset: triplet
    target: anchor_path        # structural; the loss ignores its values
    dim: 128                   # embedding dimension
```

**MULTISTREAM (dual / multi-encoder)** — N *separate* encoders (e.g. image + text), one
named stream each, aligned in a shared space. Use the `multi` backbone whose sub-encoder
names match the `data.inputs` aliases:

```yaml
backbone:
  kind: multi
  encoders:
    image: {kind: timm, name: resnet50}
    text:  {kind: timm, name: ...}      # any registered encoder

tasks:
  align:
    preset: contrastive
    target: image            # structural
    dim: 256
    loss: siglip             # swap info_nce → siglip
```

**Precomputed embeddings** — skip the image encoder entirely with the `embedding`
backbone (the input is a stored feature vector); pair it with `classification` or a metric
preset for ANN/retrieval heads.

> See `configs/experiment/{arcface,contrastive,ranking,embeddings}_smoke.yaml` for runnable
> examples. `arcface` is an angular-margin **loss** you can drop onto a `classification` task.

</details>

### Backbone

<details>
<summary>timm · smp (two feature streams) · embedding · multi-encoder; per-task <code>feature_key</code></summary>

Select the backbone group in `defaults` or override it:

```yaml
defaults:
  - backbone: resnet18    # configs/backbone/resnet18.yaml
```

| Group file | Architecture | Kind |
|---|---|---|
| `resnet18.yaml` | timm ResNet-18 | `timm` |
| `smp_unet.yaml` | smp U-Net (ResNet-34 encoder) | `smp` |
| `smp_dpt.yaml` | smp DPT | `smp` |
| `embedding.yaml` | precomputed feature vectors (no encoder) | `embedding` |

| Kind | Use for |
|---|---|
| `timm` | any timm classifier / encoder (global tasks) |
| `smp` | segmentation & multi-task (two spatial streams) |
| `embedding` | precomputed embeddings modality |
| `multi` | N named encoders for MULTISTREAM (dual-encoder / CLIP-style) |

**timm backbone** (any model from the timm registry):

```yaml
backbone:
  kind: timm
  name: efficientnet_b3
  pretrained: true
```

**smp backbone** for segmentation or multi-task:

```yaml
backbone:
  kind: smp
  name: unet
  encoder_name: resnet34
  pretrained: true
```

SMP exposes two feature streams:

| Key | Shape | Use for |
|---|---|---|
| `decoder` | `[B, D, H, W]` | segmentation head (default for `segmentation` preset) |
| `encoder_last` | `[B, D, H, W]` | classification head with SMP's internal pooling |

For **multi-task on a single smp backbone**, set `feature_key` per task:

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    # feature_key: decoder  ← default, no need to write

  label:
    preset: classification
    target: label
    class_mapping: {0: cat, 1: dog}
    feature_key: encoder_last   # explicit: use encoder output, not decoder
```

</details>

### Optimizer, LR & scheduler

<details>
<summary>Global / per-head LR, optimizer choice, and LR schedulers with runtime step counts</summary>

```yaml
lr: 1.0e-3          # global LR — all param groups start here

optimizer:
  name: adamw       # registry key: adamw · adam · sgd · rmsprop
  lr: ${lr}         # references the top-level lr
  weight_decay: 1.0e-4
```

Available optimizer groups: `adamw.yaml` · `sgd.yaml`.

**Per-head LR override**: add an `optimizer:` block to any task. That head gets its own
param group; the backbone uses the global `lr`.

```yaml
tasks:
  mask:
    preset: segmentation
    target: mask_path
    num_classes: 3
    optimizer:
      lr: 5.0e-5    # decoder head trains slower than backbone
```

**Scheduler** is its own config group (`cosine` · `onecycle` · `plateau` · `step`; `none`
= constant LR). `interval`/`frequency`/`monitor` map to Lightning's scheduling; extra keys
forward to the scheduler constructor. `runtime_kwargs` fills a constructor argument from a
trainer fact computed at fit time (`total_steps` / `steps_per_epoch` / `epochs`):

```yaml
defaults:
  - scheduler: onecycle

scheduler:
  name: onecycle
  interval: step
  max_lr: ${lr}
  runtime_kwargs: {total_steps: total_steps}   # filled from the trainer at fit time
```

```yaml
# ReduceLROnPlateau — needs a monitored metric
scheduler:
  name: plateau
  interval: epoch
  monitor: loss/val/total
  factor: 0.5
  patience: 3
```

**Per-head LR + OneCycle/Cyclic.** A scalar `max_lr` (or Cyclic's `base_lr`/`max_lr`) is
expanded per param-group, scaled by each group's lr — so a per-head `optimizer.lr` override
carries into the schedule's peak instead of being overwritten. With `max_lr: ${lr}` and a head
at `lr: 1.0e-4`, the head peaks at `1.0e-4` while the backbone peaks at `${lr}`. (`cosine` /
`step` / `plateau` already scale each group's own lr, so they need nothing special.)

</details>

### Callbacks

<details>
<summary>lr_monitor · ema · checkpoint · freeze · sample_log · batch transforms · custom</summary>

`callbacks` is a dict of `{registry_key: params}` — the same pattern as `metrics`.
Keys are looked up in `callback_registry`; values are constructor kwargs (`null` = all defaults).
Declaration order in YAML controls registration order, which matters: put `ema` before `checkpoint`.

```yaml
# configs/callbacks/default.yaml
lr_monitor:
  logging_interval: epoch

ema:
  decay: 0.999
  warmup_fraction: 0.1
  use_buffers: true

checkpoint:
  monitor: loss/val/total
  mode: min
  save_top_k: 1
  save_weights_only: true
```

Select a group in `defaults`:

```yaml
defaults:
  - callbacks: default    # lr_monitor + ema + checkpoint
  # - callbacks: minimal  # checkpoint only
  # - callbacks: none     # no callbacks (smoke tests)
```

| Key | Callback | What it does |
|---|---|---|
| `lr_monitor` | `LearningRateMonitor` | Logs learning rates to the experiment logger |
| `ema` | `EmaCallback` | Maintains an EMA shadow; validation and checkpoints use EMA weights |
| `checkpoint` | `ModelCheckpoint` | Saves the best model by a monitored metric |
| `freeze` | `FreezeCallback` | Freezes modules for the first N epochs, then unfreezes |
| `sample_log` | `SampleLogCallback` | Renders a GT-vs-prediction HTML grid (see [Sample visualization](#sample-visualization)) |
| `progress_bar` | `MetricsProgressBar` | Rich progress bar with live metrics & directions |
| `batch_transform` | `BatchTransformCallback` | Schedules MixUp / CutMix / Mosaic |

**Disable a callback at runtime** — delete its key with the `~` prefix:

```bash
uv run python main.py 'defaults=[{override /callbacks: default}]' '~callbacks.ema'
```

**Add freeze** without editing the group file — extend the dict in an experiment config:

```yaml
# configs/experiment/finetune.yaml
callbacks:
  freeze:
    targets: [model.backbone]
    unfreeze_at: 0.3    # fraction of max_epochs; int = epoch index; -1 = never
    train_bn: false
```

**Custom callback** via `_target_` (no registration needed):

```yaml
callbacks:
  my_cb:
    _target_: my_project.callbacks.GradientClipCallback
    max_norm: 1.0
```

**EMA + checkpoint**: when EMA is active, the checkpoint automatically saves EMA weights —
no special setup needed. EMA weights are swapped in before validation (where checkpoint
monitors the metric) and swapped back after.

</details>

### Logger

<details>
<summary>none (default) · ClearML</summary>

```yaml
defaults:
  - logger: none       # no logging (default)
  - logger: clearml    # ClearML experiment tracking
```

**ClearML** config:

```yaml
logger:
  kind: clearml
  project: my-project   # defaults to experiment project
  task: run-001         # optional task name
```

Override at runtime:

```bash
uv run python main.py 'defaults=[{override /logger: clearml}]'
```

</details>

### Export

<details>
<summary>ONNX / TorchScript / TensorRT with numerical-parity verification; combined & per-component graphs</summary>

After `fit`/`test`, the model is exported for deployment (gated by `run_export`). Export is
a config group (`onnx` · `torchscript` · `tensorrt` · `all`); targets are a per-format list, so one
run can emit several formats:

```yaml
defaults:
  - export: onnx        # or: torchscript · tensorrt · all

export:
  targets:
    - {format: onnx, opset_version: 17, dynamic_batch: true, simplify: true}
    - {format: torchscript, method: trace}
  combined: true          # one graph: image → all task logits
  split_components: false # also write backbone + each head as separate files
  output_dir: null        # defaults to {save_dir}/export
```

Each format validates its own option surface at `load_config` time (a misplaced
`opset_version` under `torchscript` fails immediately). Every target is **verified**: the
written artifact is re-run and its outputs compared to the source model within tolerance —

```yaml
export:
  targets:
    - {format: onnx, verify_outputs: true, atol: 1.0e-4, rtol: 1.0e-3}
```

A rich table reports per-output abs/rel error and a pass/fail verdict. Disable export with
`run_export: false` or an empty `targets` list.

**TensorRT.** The `tensorrt` target compiles straight from the PyTorch graph via torch-tensorrt
(no ONNX intermediate) to a serialized engine (`model_*.plan`) written to `{save_dir}/export/`.
The `shapes` profile references `image_size` instead of hardcoding H/W:

```yaml
defaults:
  - export: tensorrt

export:
  targets:
    - format: tensorrt
      precision: fp16          # or fp32
      atol: 1.0e-2             # fp16 needs a looser parity tolerance
      shapes:                  # min/opt/max optimization profile (drop it → batch 1/4/8)
        min: [1, 3, "${image_size.0}", "${image_size.1}"]
        opt: [4, 3, "${image_size.0}", "${image_size.1}"]
        max: [8, 3, "${image_size.0}", "${image_size.1}"]
```

> CUDA-only: a `.plan` engine is hardware + TensorRT-version specific, so build it on a node
> matching your Triton deployment. Install the optional backend once:
> `uv add --optional export-trt torch-tensorrt tensorrt`.

</details>

### Sample visualization

<details>
<summary>Interactive HTML grid of ground-truth vs predictions, via <code>sample_log</code></summary>

`SampleLogCallback` periodically takes a batch, runs the model, and renders a
self-contained interactive **HTML grid**: each cell shows the image with toggleable overlays
— chips for classification/regression labels, full-cell colored masks for segmentation —
and a sidebar to switch ground-truth / prediction layers per task and class.

```yaml
callbacks:
  sample_log:
    num_images: 8
    every_n_epochs: 5
    batch_index: 0
    title_prefix: samples
```

It is label-type agnostic: an `annotators` registry keyed by `(topology, objective)` writes
the GT/pred fields, and a `label_renderers` registry keyed by `Label` type emits the cell
overlays — so a new task kind plugs in without touching the renderer.

</details>

---

## Recipes

<details>
<summary>Single-task classification</summary>

```yaml
# configs/experiment/classify_pets.yaml
# @package _global_
defaults:
  - override /backbone: resnet18
  - override /callbacks: default

project: pets
epochs: 20
batch_size: 32
image_size: [224, 224]
lr: 1.0e-3

data:
  sources: data/pets.csv
  inputs: image_path
  split: {train: 0.8, val: 0.1, test: 0.1}

tasks:
  species:
    preset: classification
    target: species
    class_mapping: {0: cat, 1: dog, 2: rabbit}
```

```bash
uv run python main.py +experiment=classify_pets
```

</details>

<details>
<summary>Multi-task: classification + segmentation on one backbone</summary>

```yaml
# configs/experiment/multitask.yaml
# @package _global_
defaults:
  - override /backbone: smp_unet
  - override /callbacks: default

project: multitask-demo
epochs: 30
batch_size: 8
image_size: [512, 512]
lr: 1.0e-3

data:
  sources: data/annotations.csv
  inputs: image_path
  split: {train: 0.8, val: 0.1, test: 0.1}

tasks:
  mask:
    preset: segmentation
    target: mask_path
    class_mapping: {0: background, 1: defect, 2: edge}
    loss: {name: weighted_sum, losses: {cross_entropy: 1.0, dice: 1.0}}

  label:
    preset: classification
    target: label
    class_mapping: {0: ok, 1: defective}
    feature_key: encoder_last
    optimizer:
      lr: 5.0e-4
```

</details>

<details>
<summary>Metric learning: contrastive dual-encoder alignment</summary>

```yaml
# configs/experiment/align.yaml
# @package _global_
defaults:
  - override /backbone: ...        # a `multi` backbone (image + other encoder)
  - override /callbacks: default
  - override /scheduler: cosine

project: align
epochs: 50
batch_size: 64
lr: 3.0e-4

data:
  sources: data/pairs.csv
  inputs:
    image: image_path
    other: other_path
  split: {train: 0.9, val: 0.1}

tasks:
  align:
    preset: contrastive          # MULTISTREAM + InfoNCE
    target: image                # structural target
    dim: 256
    loss: siglip                 # or info_nce
```

</details>

<details>
<summary>Fine-tuning with a frozen backbone</summary>

```yaml
# configs/experiment/finetune.yaml
# @package _global_
defaults:
  - override /callbacks: default

project: finetune
epochs: 40
lr: 5.0e-4

# Extend the default callback set with freeze.
# Declare freeze before checkpoint so unfreezing runs before the save decision.
callbacks:
  freeze:
    targets: [model.backbone]
    unfreeze_at: 0.25   # unfreeze after 25% of epochs

# ... data and tasks as usual
```

</details>

<details>
<summary>Fast debugging</summary>

```yaml
# configs/experiment/debug_quick.yaml
# @package _global_
defaults:
  - override /trainer: cpu_smoke
  - override /dataloader: debug
  - override /callbacks: none
  - override /logger: none

epochs: 2
batch_size: 4
image_size: [64, 64]

data:
  max_samples: 100

# ... tasks as usual
```

</details>

---

## CLI reference

<details>
<summary>Hydra override syntax — scalars, group swaps, experiments</summary>

Override any config value directly — standard Hydra syntax:

```bash
# change a scalar
uv run python main.py epochs=5 batch_size=64

# swap a config group
uv run python main.py 'defaults=[{override /backbone: smp_unet}]'

# swap dataloader / scheduler / export presets
uv run python main.py dataloader=performance scheduler=cosine export=all

# load a full experiment override
uv run python main.py +experiment=classify_pets

# combine experiment + group swap
uv run python main.py +experiment=classify_pets 'defaults=[{override /logger: clearml}]'

# disable a callback (deletes the key from the dict)
uv run python main.py 'defaults=[{override /callbacks: default}]' '~callbacks.ema'

# per-task LR from CLI
uv run python main.py 'tasks.mask.optimizer.lr=1e-5'

# train only / eval-only / skip export
uv run python main.py run_test=false run_export=false
uv run python main.py run_train=false ckpt_path=runs/.../epoch=11.ckpt
```

</details>

---

## Extending the framework

<details>
<summary>Custom loss / metric / callback / data source via registry or <code>_target_</code></summary>

Every component is a registry key. Register your own with the `@registry.register` decorator — importing the module is enough to make it available.

**Custom loss**:

```python
# src/losses/my_loss.py
from src.losses.registry import criteria

@criteria.register("focal_tversky")
class FocalTverskyLoss(nn.Module, Criterion):
    ...
```

```yaml
tasks:
  mask:
    loss: {name: focal_tversky, alpha: 0.7, beta: 0.3}
```

**Custom metric**:

```python
from src.metrics.registry import metric_factories
metric_factories.register("my_metric")(MyTorchMetric)
```

```yaml
tasks:
  label:
    metrics:
      my_score:
        name: my_metric
        some_param: 42
```

**Custom callback**:

```python
# src/callbacks/my_callback.py
import lightning as L
from src.callbacks.registry import callback_registry

@callback_registry.register("gradient_clip")
class GradientClipCallback(L.Callback):
    def __init__(self, max_norm: float = 1.0) -> None:
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}.")
        self._max_norm = max_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        import torch
        torch.nn.utils.clip_grad_norm_(pl_module.parameters(), self._max_norm)
```

Import the module once (e.g. in `main.py`) so the decorator runs, then use it by key:

```yaml
callbacks:
  gradient_clip:
    max_norm: 0.5
```

Or use `_target_` to skip registration entirely:

```yaml
callbacks:
  my_clip:
    _target_: src.callbacks.my_callback.GradientClipCallback
    max_norm: 0.5
```

**Custom data source** (e.g. Parquet):

```python
from src.data.sources import data_sources, FileDataSource

@data_sources.register("parquet")
class ParquetDataSource(FileDataSource):
    def _read_file(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)
```

```yaml
data:
  sources: data/annotations.parquet
  source_type: parquet
```

**Other extension points** follow the same pattern — `backbones`, `head_builders`,
`target_encoders`, `input_loaders`, `topology_strategies`, `objective_strategies`,
`task_presets`, `batch_transforms`, `schedulers`, `exporters`, `label_renderers`,
`annotators`.

</details>

---

## Internals

The diagrams below show the full data flow for readers who want to understand or extend the framework internals.

<details>
<summary>Phase 1 — Setup (runs once before <code>trainer.fit</code>)</summary>

```mermaid
sequenceDiagram
    participant main
    participant Config as ExperimentConfig<br/>(Pydantic)
    participant Wiring as wiring.py
    participant DM as DataModule
    participant DS as DataSource<br/>(CsvDataSource)
    participant IL as InputLoader<br/>(per input)
    participant Codec as TargetEncoder<br/>(per task)
    participant RT as RuntimeContext
    participant TB as TaskBuilder<br/>(TopologyStrategy × ObjectiveStrategy)
    participant CM as CompositeModel<br/>(TimmBackbone + heads)

    main->>Config: load_config(raw_dict)<br/>Hydra DictConfig → validated DTOs
    Config-->>main: ExperimentConfig

    main->>Wiring: build_bindings(config)
    Note over Wiring: infers TargetEncoder per task<br/>from preset + objective
    Wiring-->>main: list[TargetBinding(name, column, encoder)]

    main->>Wiring: build_data_module(config, target_bindings, runtime)
    Note over Wiring: resolves split/pre-split mode,<br/>builds transforms from configs/transforms/
    Wiring-->>main: DataModule

    main->>DM: setup()

    DM->>DS: read()
    DS-->>DM: DataFrame (full annotation table)

    DM->>DM: _build_input_bindings(inputs_config, frame)
    Note over DM: auto-detects loader per column<br/>from file extensions → image/text
    DM-->>IL: InputBinding(name, column, loader) per input

    loop for each TargetBinding
        DM->>Codec: fit(column_values)
        Note over Codec: LabelEncoder → sorts vocab<br/>MultiLabelEncoder → multi-hot vocab<br/>ScalarEncoder / MaskEncoder → no-op
        Codec-->>RT: num_classes[task_name] = C
    end

    DM->>DM: split_dataframe(ratios, seed, stratify_column?)
    Note over DM: → Dataset per stage<br/>(frame + input_bindings + target_bindings + transform)
    DM-->>RT: dataset_sizes[stage] = N

    main->>Wiring: build_tasks(config, runtime)
    Note over Wiring: runs AFTER setup()<br/>so num_classes is a concrete int

    loop for each task in config.tasks
        Wiring->>TB: build(name, num_classes, objective, ...)
        TB->>TB: validate Topology × Objective
        TB->>TB: out_features / head_spec / adapter<br/>criterion / activation / metrics × stages
        TB-->>Wiring: Task { head_spec, adapter, criterion,<br/>activation, metrics, weight }
    end
    Wiring-->>main: list[Task]

    main->>CM: build_composite_model(backbone, {name: HeadSpec})
    Note over CM: TimmBackbone + nn.ModuleDict(head per task)<br/>head sized from backbone.feature_dim(feature_key)
    CM-->>main: CompositeModel

    main->>main: build_lit_module(...) ; build_lit_data_module(...)<br/>build_trainer(...) ; run_experiment(fit → test → export)
```

</details>

<details>
<summary>Phase 2 — Training step (repeats every batch)</summary>

```mermaid
sequenceDiagram
    participant Trainer as L.Trainer
    participant LitM as LitModule
    participant DL as DataLoader
    participant DS as Dataset.__getitem__
    participant CM as CompositeModel
    participant BB as TimmBackbone
    participant Head as LinearHead / ConvHead<br/>(per task)
    participant TC as TargetAdapter<br/>(per task)
    participant Crit as Criterion<br/>(per task)
    participant Act as Activation<br/>(per task)
    participant Met as MetricSet<br/>(per task × stage)
    participant Agg as WeightedSumAggregator

    Trainer->>LitM: training_step(batch, batch_idx)

    LitM->>DL: next(dataloader)
    DL->>DS: __getitem__(index)
    loop for each InputBinding
        DS->>DS: InputLoader.load(value) → ndarray / str
    end
    DS->>DS: Transform.apply(sample) → tensors [C,H,W]
    loop for each TargetBinding
        DS->>DS: encoder.load(raw) then encoder.to_tensor(val) → Tensor
    end
    DS-->>DL: Sample { inputs: {alias: Tensor}, targets: {task: Tensor} }
    DL->>DL: collate_samples(list[Sample])
    DL-->>LitM: Batch { inputs: {…: [B,C,H,W]},<br/>targets: {task_name: Tensor[B,…]} }

    LitM->>CM: forward(batch.inputs)
    CM->>BB: forward({"image": Tensor[B,C,H,W]})
    BB-->>CM: FeatureBundle { "pooled": [B,D] / "decoder": [B,D,H,W] }

    loop for each task head
        CM->>Head: forward(features[task.feature_key])
        Head-->>CM: logits Tensor[B, out_features]
    end
    CM-->>LitM: ModelOutput { task_logits: {name: Tensor} }

    loop for each Task
        LitM->>TC: adapt(batch.targets[task.name])
        Note over TC: MulticlassTargetAdapter → [B] long<br/>BinaryTargetAdapter    → [B,1] float / [B,1] long<br/>MultilabelTargetAdapter→ [B,C] float / [B,C] long<br/>ContinuousTargetAdapter→ [B,1] float
        TC-->>LitM: TargetView { loss: Tensor, metric: Tensor }

        LitM->>Crit: forward(logits, target.loss)
        Note over Crit: CrossEntropyCriterion / BCEWithLogitsCriterion<br/>MSECriterion / DiceCriterion / CompositeCriterion
        Crit-->>LitM: LossResult { total, components: {name: Tensor} }

        LitM->>Act: __call__(logits)
        Note over Act: SoftmaxActivation / SigmoidActivation / IdentityActivation
        Act-->>LitM: preds Tensor

        LitM->>Met: update(preds, target.metric)
        Note over Met: TorchMetricsAdapter wraps MetricCollection<br/>accumulates state across batches
    end

    LitM->>Agg: combine(losses, weights)
    Note over Agg: total = Σ weight_i × task_i.total<br/>components namespaced as "task/component"
    Agg-->>LitM: LossResult { total (scalar), components }

    LitM->>LitM: self.log("loss/train/total", …)
    LitM-->>Trainer: combined.total  →  .backward()
```

</details>

<details>
<summary>Epoch end — metrics flush</summary>

```mermaid
sequenceDiagram
    participant Trainer as L.Trainer
    participant LitM as LitModule
    participant Met as MetricSet<br/>(per task × stage)
    participant Log as Lightning Logger

    Trainer->>LitM: on_train_epoch_end() / on_validation_epoch_end()

    loop for each Task
        LitM->>Met: compute()
        Met-->>LitM: { "accuracy": 0.83, "macro_f1": 0.79, … }
        LitM->>Log: self.log("{task}/{metric}/{stage}", value)
        LitM->>Met: reset()
    end
```

</details>
