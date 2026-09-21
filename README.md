# ml-framework

Config-driven multi-task computer-vision training on PyTorch Lightning, Hydra and
Pydantic, with timm, segmentation-models-pytorch, albumentations and torchmetrics behind
it.

**Who this is for.** Data scientists who want to train, evaluate and ship a vision model
by writing a YAML file instead of a training loop. You need Python and PyTorch. You do
not need to read the framework's source to use it — and you do not need to edit it to
extend it.

**What you can train.** Classification (single-label, binary, multi-label), semantic
segmentation, regression, metric learning and contrastive pretext runs — several of them
on one backbone at once. Beside that: EMA, layer freezing, MixUp and CutMix, LoRA
adapters, loss-parameter annealing, per-task learning rates, knowledge distillation from
a second network, a grid of sample predictions and a summary of the data a run is about
to read.

**What a run ends with.** A deployable artifact — ONNX, PT2, TorchScript, ncnn or a
TensorRT engine — each checked against the model it was written from and described by a
`model.json` a deployment reads.

**Not covered.** Object detection: no box geometry, encoders or metrics for it.

## Quick start

```bash
make install
make test-run
```

`make test-run` downloads the Oxford-IIIT Pet dataset, writes its table, and trains
classification, regression and segmentation together on one backbone. It is the whole
framework in one command.

Every shipped example is one line from there:

```bash
uv run main.py experiment=examples/classification
uv run main.py experiment=examples/segmentation
uv run main.py experiment=examples/metric_learning
uv run main.py experiment=examples/finetuning
uv run main.py experiment=examples/classification lr=3e-4 epochs=50 scheduler=onecycle
```

The examples live in [`configs/experiment/examples/`](configs/experiment/examples/) and
a test assembles every one of them, so none can drift from the code:

| Example | What it runs |
|---|---|
| `classification.yaml` | Cat or dog, from a pretrained ResNet |
| `segmentation.yaml` | Pet, background and boundary, per pixel |
| `regression.yaml` | A continuous target |
| `multitask.yaml` | All three at once, on one backbone |
| `metric_learning.yaml` | Identities rather than classes, judged on breeds held out of training |
| `finetuning.yaml` | Freeze, release, average: the classic pretrained-backbone recipe |
| `caption.yaml`, `pairing.yaml` | Text, and image-text pairs |
| `pet.yaml` | The table and split the others share; inherited, never run on its own |
| `contrastive.yaml` | A pretext run. Known not to assemble; tracked as a defect |

## How a run is declared

Four ideas carry the rest of this page.

**One experiment file is one run.** It names the data, the tasks and whatever it wants
different from the defaults. Everything it does not name comes from
[`configs/config.yaml`](configs/config.yaml) and the group files beside it.

**Every component is declared the same way.** `name` picks a registered implementation;
`_target_` takes an import path to anything, including your own class. Every other key in
the mapping is an argument for that component's constructor, so an upstream knob is
reachable without a schema change:

```yaml
loss: {name: focal, gamma: 2.0}              # registered by name
loss: {_target_: my_project.losses.Tversky}  # anything importable
loss: cross_entropy                           # shorthand for {name: cross_entropy}
```

**Sizes come from the data.** Encoders fit on the training split, and heads are built
from what they found. You never write `num_classes`, an embedding width the data settles,
or a feature dimension.

**A bad declaration dies while the run is assembled**, with a message naming the key and
the fix — not mid-epoch, and never by one value silently winning over another.

### Overriding from the command line

Change a knob at its one home, not at a mirror of it:

```bash
uv run main.py experiment=examples/classification lr=3e-4 epochs=50
uv run main.py experiment=examples/classification model=unet tracker=csv export=onnx
uv run main.py experiment=examples/classification +trainer.precision=bf16-mixed
```

`lr` lives in the root config and the optimizer group interpolates from it, so
`optimizer.lr=3e-4` is refused. Use `+` only for a key a group genuinely does not
declare, such as a Lightning `Trainer` argument.

### What each section decides

| Section | Decides | Default |
|---|---|---|
| `data` | The table, which columns are inputs, how it is split | none — you declare it |
| `tasks` | What is learned, from which column, judged by what | none — you declare it |
| `model` | The backbone the heads are built onto | `resnet18` |
| `preprocessing` | How a sample is loaded and normalized | `image` |
| `transforms` | The pixel chain, per split | `default` (deterministic) |
| `optimizer`, `scheduler` | How the weights move | `adamw`, `none` |
| `trainer`, `loader` | Lightning and DataLoader arguments, forwarded verbatim | `default` |
| `callbacks` | Checkpoints, EMA, freezing, mixing, reports | `none` |
| `tracker` | Where numbers are recorded | `none` |
| `export` | What is shipped when the run ends | `none` |
| `adapter` | Parameters added to a frozen part, then folded back | `none` |
| `learner` | The training algorithm, and a teacher where one learns from a second network | `standard` |
| `run` | Project and run name, directories, what to restore, whether to train or only test | — |

`model`, `optimizer`, `scheduler`, `trainer`, `loader`, `callbacks`, `tracker`, `export`,
`adapter`, `preprocessing` and `transforms` are Hydra groups: swap one with
`model=unet`. A group replaces a list rather than appending to it, so a run that declares
its own `callbacks` writes the whole list.

## Config recipes

Terse, working fragments. Each one goes in an experiment file.

<details>
<summary><b>The smallest run</b></summary>

```yaml
# @package _global_
data:
  source: data/train.csv
  inputs: {image: {column: image_path}}
  split: {train: 0.8, val: 0.2}

tasks:
  label:
    kind: classification
    target_column: species
    classes: {0: cat, 1: dog}
```

`classes` is declared rather than inferred: the data is validated against this vocabulary,
the index space survives resampling, and these names label the metrics, the confusion
matrix and the samples grid.
</details>

<details>
<summary><b>Segmentation</b></summary>

```yaml
defaults:
  - override /model: unet

tasks:
  mask:
    kind: segmentation
    target_column: mask_path
    classes: {0: pet, 1: background, 2: boundary}
    head: native          # smp's own segmentation head, over the decoder
    loss:
      - {loss: cross_entropy, weight: 1.0}
      - {loss: dice, weight: 1.0}
```

Several losses on one output: each entry names its loss apart from its mixing weight, so
a loss's own `weight` argument never collides with it.
</details>

<details>
<summary><b>Several tasks on one backbone</b></summary>

```yaml
tasks:
  mask:
    kind: segmentation
    target_column: mask_path
    classes: {0: pet, 1: background, 2: boundary}
    head: native
    weight: 1.0
  label:
    kind: classification
    target_column: species
    classes: {0: cat, 1: dog}
    head: {name: native, stream: encoder}   # a linear head cannot read a feature map
    weight: 0.5
    lr: 1.0e-3                              # its own pace: small head, pretrained trunk
  age:
    kind: regression
    target_column: age
    head: {name: native, stream: encoder}
    weight: 0.2
```

`weight` is the task's share of the total objective; `lr` gives it its own optimizer
group, which `lr_monitor` then draws as its own line.
</details>

<details>
<summary><b>Metric learning</b></summary>

```yaml
data:
  split:
    rule: {_target_: src.data.GroupedSplit, by: breed}   # whole identities held out

tasks:
  identity:
    kind: {name: metric_learning, embedding_dim: 128}
    target_column: breed
    loss: {name: arcface_proxy, margin: 0.5, scale: 64.0}
    metrics:
      recall_at_1: {name: recall_at_k, k: 1}
      map: {name: map}
      verification: {name: verification_accuracy}
      threshold: {name: verification_threshold}

callbacks:
  - {name: checkpoint, monitor: val/identity/recall_at_1, mode: max, dirpath: ${run.directory}/checkpoints}
```

No `classes`: this kind learns its vocabulary from the training split alone, so the
identities held out are not held to a list. `val/loss` is the wrong monitor here — the
objective is over the training identities, and validation names others.

The other arrangement puts the prototypes in the network and classifies directly:

```yaml
tasks:
  breed:
    kind: classification
    target_column: breed
    classes: {...}
    head: {name: cosine, embedding_dim: 128}
    loss: {name: arcface, margin: 0.5, scale: 64.0}
```
</details>

<details>
<summary><b>Distillation from a second network</b></summary>

```yaml
learner:
  name: distillation
  weight: 1.0                                        # the teacher's share, beside the tasks' own
  loss: {name: kullback_leibler, temperature: 4.0}
  teacher:
    name: composite
    backbone: {name: timm, model_name: vit_large_patch16_dinov3.lvd1689m}
    checkpoint_path: runs/teacher/checkpoints/best.ckpt
```

The teacher is sized by this run's own tasks, so its widths are never written twice. It is
held outside the module tree: no checkpoint carries it, no export ships it. Declare
`learner.teacher.heads` where it should reach those widths through a head of its own — see
*Continuing a teacher's head*.

A head answering in cosines needs `scale`, which turns a bounded answer into a
distribution before it is softened. Declared without one, the pair is refused by name:

```yaml
  loss: {name: kullback_leibler, temperature: 4.0, scale: 16.0}
```

Watch `train/<task>/distillation`. The column holds `weight × temperature² × KL`, so
divide by `temperature²` to compare runs at different temperatures.
</details>

<details>
<summary><b>Continuing a teacher's head</b></summary>

A teacher trained with a wide head leaves a stack of layers. Where the student's backbone
publishes the width that stack passes through, the student can carry the tail of it:

```yaml
# The teacher: the whole head, over its own 1024-wide features.
learner:
  name: distillation
  weight: 1.0
  loss: {name: kullback_leibler, temperature: 4.0}
  teacher:
    name: composite
    backbone: {name: timm, model_name: vit_large_patch16_dinov3.lvd1689m}
    checkpoint_path: runs/teacher/checkpoints/best.ckpt
    heads:
      species: {name: mlp, hidden_features: [1280, 128, 64]}

# The student: the tail alone, over its own 1280-wide features, held still.
tasks:
  species:
    head: {name: mlp, hidden_features: [128, 64], checkpoint_path: weights/tail.pt}

callbacks:
  - {name: freeze, modules: [heads.species]}
```

`mlp` puts a GELU between every pair of projections. Left without `hidden_features` it is
one layer as wide as what it reads; declared empty it is refused, because a head of one
projection is `linear`.

A head's `checkpoint_path` holds weights for **exactly that head** — the same names at the
same widths — and the file extension is not read, only its contents. A whole run's file is
refused by name, because its tensors are the run's rather than the head's. Cut the tail out
of one with
[`scripts/cut_head_tail.py`](scripts/cut_head_tail.py). Given the widths the student's head
is built at, it renumbers the layers and checks the cut before it writes:

```bash
uv run python -m scripts.cut_head_tail runs/teacher/checkpoints/best.ckpt weights/tail.ckpt \
    --task species --in-features 1280 --out-features 8 --hidden 128 64
```

Given no widths at all, it takes the head whole — which is what a student carrying all of
its teacher's head needs, and the ordinary case once both networks are brought to one width:

```bash
uv run python -m scripts.cut_head_tail runs/teacher/checkpoints/best.ckpt weights/head.ckpt \
    --task species
```

**What this arrangement transfers, and what it does not.** A frozen head constrains the
route the student takes to its answer, not the answer itself: the backbone beneath it is
free to produce whatever features make the logits come out right. Against `kullback_leibler`
over logits alone, freezing the tail therefore changes nothing that a trainable head would
not also reach. What the arrangement buys is a shared space to compare *features* in — both
networks reach the same widths by construction, so no neck is needed. The term that
reads those features is a `stream` on `learner.loss` — see below.
</details>

<details>
<summary><b>A neck between the backbone and its heads</b></summary>

`model.neck` is the position between the two: a backbone reads a sample, a neck reads the
features it published, and the heads are sized from whichever of them published last. Left out
— which is every ordinary run — the heads read the backbone and nothing changes, not even the
checkpoint the run writes.

`projector` republishes one stream through a single linear layer, at a width the run writes
down. Nothing downstream moves: the stream keeps its name, so a head reads `pooled` as it
always did and is sized from the new number without being told it.

```yaml
model:
  name: composite
  backbone: {name: timm, model_name: mobilenetv4_conv_small}
  neck: {name: projector, width: 128}
```

Measured on exactly that declaration: the backbone publishes `pooled` at 1280 and the neck
publishes it at 128, through `Linear(1280, 128)`.

**Beside the backbone, not around it.** The paths a composite registers — `backbone`, `neck`,
`heads.<task>` — are what `freeze`, `adapter` and every checkpoint address, so a neck leaves
them where they were: `modules: [backbone]` holds the encoder and lets the projection learn,
which is the recipe `examples/finetuning.yaml` ships. A projection declared *inside* the
backbone would be held still along with it, and the run would report a frozen encoder while
the one layer it exists to train never moved.

One linear layer, and no knob for a second — a stack of projections is what a head is, and
`mlp` is where a run declares one.

`stream` is left out where the backbone publishes one, and named where it publishes several:
`multiencoder` publishes one per tower (`image_pooled`, `text_pooled`). Streams the neck does
not bring are published exactly as they arrived. A stream that is still spatial — what `smp`
publishes — is refused by name, because a projection reads a pooled vector and a feature map
is brought to a width by a convolution this neck does not build.

Per stream, and not per run: over a stream the neck brought, `head: native` is refused — the
library's own classifier reads a feature space that is gone — and a classifier a
`checkpoint_path` carried has nowhere to land, so that head starts fresh and the run says so
rather than dropping a warm start in silence. Over a stream the neck passed through, both are
exactly what they were.

**What this is for.** Two networks meeting in one space. A teacher and a student publish
whatever widths their libraries chose, and a neck on either brings both to one declared width;
from there a head trained on one of them reads the other, and a term of `learner.loss` naming
that stream is what pulls one towards the other.
</details>

<details>
<summary><b>Distilling features as well as answers</b></summary>

`learner.loss` takes one term or a weighted list of them, the way `tasks.<name>.loss` does. A
term that names a `stream` compares that feature of the two networks; a term that names none
compares their answers.

```yaml
learner:
  name: distillation
  weight: 1.0                                   # what the teacher is worth in all
  loss:
    - {loss: {name: kullback_leibler, temperature: 3.0}}
    - {loss: mse, weight: 5.0, stream: pooled}  # its share within that
  teacher:
    name: composite
    backbone: {name: timm, model_name: vit_large_patch16_dinov3.lvd1689m, pretrained: false}
    checkpoint_path: runs/teacher/checkpoints/best.ckpt
```

The terms report as `<task>/distillation` and `<stream>/representation`, each named for what it
*read* rather than for the loss it used, so a column survives a change of measure; two terms over
one reading need telling apart, and `log_name` is how a run does it.

`learner.weight` is what
 everything learned from the teacher is worth beside the tasks' own
objectives, and the weight inside a term is its share of that — the two levels a task and its
loss list already have.

Both networks have to publish the named stream at the same shape, or the term is refused by
name with both shapes shown. Where they already publish the same width, no neck is wanted;
where they do not, a `model.neck` on either brings them to one.

**Where the two networks meet.** Three arrangements, one declaration:

| | neck | the space they share | what it costs |
|---|---|---|---|
| on the student alone | student → teacher's width | the teacher's own features | one wide layer, and the head reads a space it was never narrowed for |
| their widths already agree | on neither | whatever both publish | nothing |
| on both | each → a width you chose | that width, which the teacher trained for the task | two narrow layers, and a teacher trained with its neck |

The first asks nothing of the teacher: an already-trained one is used as it stands, so it is
the cheapest thing to try, and it is written out below. The last is the cheapest at inference
and the most task-specific, because every direction in a narrow space the teacher trained is a
direction its head reads.

That first arrangement whole, with the teacher's head cut out by the command above and carried
by the student, which is the pair that makes any of this mean something:

```yaml
model:
  name: composite
  backbone: {name: timm, model_name: mobilenetv4_conv_small}
  neck: {name: projector, width: 1024}           # what the teacher publishes, and now the student too

tasks:
  species:
    head: {name: linear, checkpoint_path: weights/head.ckpt}   # the teacher's own, cut out whole

learner:
  name: distillation
  weight: 1.0
  loss:
    - {loss: {name: kullback_leibler, temperature: 3.0}}
    - {loss: mse, weight: 5.0, stream: pooled}
  teacher:
    name: composite
    backbone: {name: timm, model_name: vit_large_patch16_dinov3.lvd1689m}
    checkpoint_path: runs/teacher/checkpoints/best.ckpt

callbacks:
  - {name: freeze, modules: [heads.species]}     # what makes the shared space mean anything
```

The other two are that one with a line moved. **Where the widths already agree**, drop the
`neck` line — nothing else changes. **A neck on both** gives the teacher one too, at whatever
width you choose, and the teacher is trained that way before its head is cut out: the head then
reads the narrow space rather than the wide one, and `width` is the same number on both sides.
`learner.teacher` is an ordinary model declaration, so it takes `neck` by the same word.

Each of the three is assembled through the real composition root in
[`tests/e2e/test_distillation_run.py`](tests/e2e/test_distillation_run.py) — which is where the
column names above come from.


**Why both terms.** Answers carry the proportions a label leaves out — how wrong each of the
other classes is — but a confident teacher spends nearly all of that on one class, leaving
little there to learn. Features carry the representation those answers were read from. With
the teacher's head carried over and held still, a student whose features land where the
teacher's do answers as the teacher does, by construction — and that is what makes freezing
the head worth anything, since on its own, under a divergence over logits, it constrains
nothing a trainable head would not also reach.
</details>

<details>
<summary><b>LoRA</b></summary>

```yaml
defaults:
  - override /adapter: lora
  - override /model: dpt_dinov3

adapter:
  module: backbone.encoder     # what is pretrained, and nothing else
  target_modules: [qkv, proj]  # matches the end of a module path
  r: 8
  alpha: 16
```

`target_modules` reaches every block whose path ends in that word: `qkv` and `proj` are a
transformer's attention, `fc1` and `fc2` its MLP, `conv1` and `conv2` a residual
network's convolutions. A name reaching nothing is refused while the run is assembled.

The delta is folded into the weights after the kept epoch is restored, so `state_dict`
keys, `model.json` and every export read exactly as in a run that adapted nothing.
</details>

<details>
<summary><b>Freezing, EMA, mixing, annealing</b></summary>

```yaml
callbacks:
  - {name: progress}
  - {name: model_summary, max_depth: 3}
  - {name: lr_monitor, logging_interval: epoch}
  - {name: metric_summary}

  # Hold the backbone still while the fresh head stops pushing noise into it,
  # then let it go a third of the way in.
  - {name: freeze, modules: [backbone], until: 0.3}

  # Average the last stretch of weights; validate and save in their place.
  - {name: ema, decay: 0.999, after: 0.3}

  # Two samples become one picture and one blended label. Stopped before the end,
  # so the last epochs are spent on the data the run is judged on.
  - name: batch_transform
    transform: {_target_: src.transforms.MixUp, alpha: 0.4}
    until: 0.8

  # Move one number of a task's objective over the run.
  - {name: anneal, task: label, parameter: focal.gamma, start: 0.0, end: 2.0}

  - name: checkpoint
    monitor: val/loss
    mode: min
    save_top_k: 1
    dirpath: ${run.directory}/checkpoints
```

`until` and `after` are a share of the run (`0.3`) or an epoch number (`3`). `1` is
refused as ambiguous; leave the knob out to mean the whole run.

EMA needs a full checkpoint. Declaring `save_weights_only: true` beside it is refused,
because the file would hold the live weights while the metric that chose it came from the
averaged ones.
</details>

<details>
<summary><b>Splitting a table</b></summary>

```yaml
data:
  split:
    train: 0.7
    val: 0.15
    test: 0.15
    seed: 42
    # Leave `rule` out to shuffle and cut.
    rule: {_target_: src.data.StratifiedSplit, by: species}   # keep each split's class mix
    # rule: {_target_: src.data.GroupedSplit, by: patient_id} # keep a group on one side
```

`seed` is separate from the experiment's, so runs at different seeds can share a split.
</details>

<details>
<summary><b>Exporting and tracking</b></summary>

```bash
uv run main.py experiment=examples/classification export=onnx tracker=clearml
```

```yaml
export:
  - {name: onnx, opset: 18, simplify: true}
  - {name: torchscript}
```

`export=all` writes ONNX, PT2 and TorchScript. `ncnn` converts through `pnnx` from a
TorchScript graph, and `tensorrt` compiles the ONNX graph for one GPU — it needs the
`tensorrt` package, which is not a dependency here and needs a GPU at import.

Each artifact is checked against the model it came from and described by a `model.json`
beside it.

Trackers: `none` (default), `csv` for a local file, `clearml` to upload. `lr_monitor`
needs one; `metric_summary` adds its table only where a backend keeps one, and is left
alone otherwise.
</details>

<details>
<summary><b>Starting from weights, resuming, testing only</b></summary>

```yaml
run:
  checkpoint_path: runs/previous/checkpoints/best.ckpt   # weights only
  # resume_path: ...                                     # weights + optimizer + epoch
  train: false                                           # score an existing model
  test: true
```

Declare one of `checkpoint_path` or `resume_path`, not both. To load weights for a
*backbone architecture* rather than a whole run, use `model.backbone.checkpoint_path`
instead.
</details>

## Extending it

Nothing under `src/` has to change. A class of your own is reached by `_target_` wherever
a `name` would go, and receives the same derived facts a registered one does.
[`tests/e2e/test_custom_extension.py`](tests/e2e/test_custom_extension.py) runs a task
kind, a network and a splitting rule of one's own, end to end.

<details>
<summary><b>A loss, metric, callback or encoder of your own</b></summary>

Write the class, then name it:

```python
# my_project/losses.py
from torch import nn, Tensor


class Ranking(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor: ...
```

```yaml
tasks:
  label:
    loss: {_target_: my_project.losses.Ranking, margin: 0.3}
```

Any module comparing two tensors works as a loss. A constructor that names a fact the run
derived — `num_classes`, `semantics` — is handed it; one that does not, is not.

The same holds for a metric (anything with torchmetrics' interface), a callback (a
Lightning `Callback`), a target encoder and a pixel transform.
</details>

<details>
<summary><b>A backbone, or a whole model</b></summary>

What the third-party model gives you decides what you write, and each row is one new
class:

| The model provides | You write |
|---|---|
| Features only (timm, DINO, an smp encoder) | a `Backbone` adapter |
| Features behind a removable head (torchvision) | a `Backbone` adapter that strips it |
| Logits but no loss (a Hugging Face `*ForClassification`) | a `Backbone` exposing a `logits` stream, plus `head: {_target_: torch.nn.Identity}` |
| Everything — head, loss, decoding (a DETR of your own) | a `Model`, reached by `_target_` and built as it is |
| Only weights for a topology this framework already builds | nothing; `run.checkpoint_path` loads them |

```yaml
model: {_target_: my_project.models.Detector, variant: small}
```

A model reached this way arrives whole: no backbone is composed and no heads are built
onto it. If it answers in something other than raw projections, say so by overriding
`produces` — the objective over it is checked against that answer.
</details>

<details>
<summary><b>A kind of task</b></summary>

A kind states, in one class, what a task of that kind needs: which encoder reads its
column, which head serves it, what it is judged by, and how its output is read back.

```python
from src.tasks import Task


class Doubling(Task):
    default_target_encoder = "scalar"
    ...
```

```yaml
tasks:
  age: {kind: {_target_: my_project.tasks.Doubling}, target_column: age}
```

Subclass `Task` and read the contract in
[`src/tasks/base.py`](src/tasks/base.py); the shipped kinds beside it are the worked
examples.
</details>

<details>
<summary><b>A rule for dividing a table</b></summary>

A rule is a callable taking the rows, the fraction wanted per split name, and a seed:

```python
class EveryOther:
    def __init__(self, first: str = "train") -> None:
        self.first = first

    def __call__(self, rows, fractions, seed): ...
```

```yaml
data:
  split: {train: 0.5, val: 0.5, rule: {_target_: my_project.splits.EveryOther}}
```
</details>

<details>
<summary><b>Where new code belongs</b></summary>

Dependencies point in one direction: a thin core of entities and registries, capability
packages around it, and one composition root that alone reads the whole declaration. The
core never imports a capability, a capability never imports `config/`, and each
third-party stack stays in one package — Lightning in `training/` and `callbacks/`,
albumentations in `transforms/`, pandas in `data/`, ONNX and TensorRT in
`export/backends/`.

This is enforced, not advised:
[`tests/test_layering.py`](tests/test_layering.py) walks every import in the tree and
fails on one that is not declared.
</details>

## What ships

Names usable as `name:` in their own position.

| Position | Names |
|---|---|
| `tasks.<n>.kind` | `classification`, `binary_classification`, `multilabel_classification`, `segmentation`, `binary_segmentation`, `regression`, `metric_learning`, `contrastive` |
| `tasks.<n>.loss` | `cross_entropy`, `bce`, `focal`, `dice`, `iou`, `tversky`, `mse`, `mae`, `huber`, `smooth_l1`, `expectation`, `arcface`, `arcface_proxy`, `info_nce` |
| `tasks.<n>.metrics` | `accuracy`, `f1`, `precision`, `recall`, `iou`, `mae`, `mse`, `confusion_matrix`, `recall_at_k`, `map`, `verification_accuracy`, `verification_threshold` |
| `tasks.<n>.head` | `linear`, `cosine`, `conv`, `mlp`, `native` |
| `tasks.<n>.target_encoder` | `label`, `identity`, `scalar`, `multilabel`, `mask`, `linear_bins`, `gaussian_bins` |
| `model.backbone` | `timm`, `smp`, `hf_text`, `multiview`, `multiencoder` |
| `model.neck` | `projector` |
| `callbacks` | `checkpoint`, `progress`, `model_summary`, `metric_summary`, `lr_monitor`, `ema`, `freeze`, `batch_transform`, `anneal`, `samples`, `dataset_summary` |
| `learner` | `standard`, `distillation` |
| `learner.loss` | `kullback_leibler`, `mse`, `mae` — one term, or a weighted list of them |
| `adapter` | `lora` |
| `export` | `onnx`, `pt2`, `torchscript`, `ncnn`, `tensorrt` |
| `tracker` | `csv`, `clearml` |
| `data.source.format` | `csv`, `json`, `jsonl` |

A misspelled name is refused with the list of the ones that position holds.

## Development

```bash
make install     # sync dependencies
make test        # the whole suite
make test-unit   # one package at a time
make test-e2e    # assembled runs, end to end
make test-gate   # the suite minus the tests that need a model hub
make typecheck   # mypy over src and tests
make check       # typecheck + tests
make pre-commit  # every hook: file hygiene, typos, ruff, mypy, the test gate
make clean       # caches and temporary files
```

Every contract is stated in the docstring of the `base.py` that declares it, and the
config files under [`configs/`](configs/) carry the reasoning behind their defaults.
