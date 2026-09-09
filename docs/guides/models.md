# Models

The `model:` section is one component like any other: a registry name and its
constructor arguments, every upstream knob reachable verbatim:

```yaml
model:
  name: timm
  model_name: convnext_tiny
  drop_rate: 0.1            # forwarded to timm.create_model untouched
```

Heads are never configured here — they derive from tasks and size themselves
from the task's facts.

## Detection on a composed backbone

`model: {name: ultralytics, model_name: yolov8n.yaml}` builds ultralytics' graph and
serves it in the composite grammar: everything before `Detect` is the backbone, exposing
the pyramid `p3`, `p4`, `p5` (strides 8/16/32) and `features` — P5 pooled to `[B, D]` —
so a whole-image task sits beside the boxes on one backbone (`streams: features`).
A `detection` task reads the pyramid the backbone declares — three levels for `yolov8n`,
four for `yolov8n-p2`, named by stride — and the framework has no detection head of its
own, so the backbone's `Detect` is rebuilt per task at the task's class count
(`head: native` is the default there, not a knob). The head returns one raw tensor
`[B, 4·reg_max + nc, A]` in every mode; decoding is a separate step.

Every yaml ultralytics ships whose head is the one-to-many `Detect` is a config string
(v8, 11, 12); `v10Detect`, RT-DETR and the end-to-end `Detect` of yolo26 (`end2end: True`,
`reg_max: 1`) are refused by name until their own head adapters land.
Weights never come with the name: `checkpoint_path: runs/yolov8n.pt` grafts an
ultralytics `.pt` — the body strictly, the head's box branch always, its class branch
when the class count matches — and says what it transplanted. The file is a pickled
module, read with `weights_only=False`: only a path you wrote is ever unpickled.

Until stage 3 of the detection roadmap a composed detection model builds but has no
criterion; a run declaring one builds the model and is refused before training, naming
the stage.

## A model that arrives whole

Everything above composes: this framework wraps a backbone, builds the heads, declares
the criteria. A model may instead arrive whole — its head, its loss and its decoding one
design of its own — and it is reached like any component:

```yaml
model:
  _target_: my_pkg.MyDetector   # a `Model`: step() and predict() are its own
  num_classes: 3                # what it needs, it declares — nothing composes it
```

`build_model` takes a built `Model` as it is; a `Backbone` is what gets heads composed
onto it, and anything else is refused naming both. The run around it is unchanged: the
table pipeline feeds it, the tasks' kinds judge it with their metrics, its `Loss.parts`
log under `{stage}/{task}/{part}`, checkpoints and export work through the same ports.

**What such a run refuses, at build time, by name:** `adapters` (they reparameterize a
backbone this framework composed) and `distillation` (it compares composed per-task
logits). A per-task `lr` is refused too, by the mechanism that already exists for it:
the model exposes no per-task parameters, so a declared rate has nothing to move.

## Arrived weights

Trained weights of a timm architecture often arrive as a file. `checkpoint_path`
loads them with `timm.load_checkpoint`'s own knob names and defaults, so the
options read exactly as a DS knows them:

```yaml
model:
  name: timm
  model_name: convnext_tiny
  checkpoint_path: runs/v1/model.pt
  # use_ema: true          # prefer the checkpoint's EMA branch when present
  # strict: true           # false allows deliberate partial loads, reported loudly
  # weights_only: true     # torch.load safety knob
```

What the framework adds on top of timm: the run logs how many tensors loaded,
warns with the exact missing/unexpected key lists on a partial load, and
refuses a checkpoint that matches nothing — a weight file never no-ops
silently. `pretrained` is moot alongside `checkpoint_path`: a file is the
weight source, so the hub is not downloaded just to be overwritten.

The checkpoint's classifier does not load into the backbone (the backbone is
headless) — it is stashed for the task's head, below.

## Growing the class space

The recurring fine-tune: a trained multilabel model must learn one novel
class, with new data labelled only for it. The old classifier rows must
survive verbatim — there is no data to relearn them:

```yaml
model:
  name: timm
  model_name: convnext_tiny
  checkpoint_path: runs/v1/model.pt

tasks:
  tags:
    kind: multilabel_classification
    target: tags
    head: native
    classes: {0: indoor, 1: outdoor, 2: people, 3: night}   # night is novel

callbacks:
  - name: freeze
    modules: [backbone, heads.tags.base]
    until: 0.5              # release halfway through the run; omit to keep frozen
```

`head: native` asks the backbone for its own classifier, and a timm
backbone with a stashed checkpoint transplants it: with 3 carried rows and 4
declared classes the head becomes two named parts — `base`, the transplanted
`[3, D]` classifier, and `novel`, a fresh `[1, D]` block — concatenated on
forward. The names are the freeze-path contract: `heads.<task>.base`
freezes the carried rows at a module boundary, which is the only honest way
(a gradient mask would still let AdamW's weight decay move them).

The declared `classes` carry the index contract: old classes must keep their
old indices, novel ones follow — visible in the config, validated where the encoder is built.

The same story holds for segmentation with the smp family — the knobs, the
report, and the growth are identical; per-class channels of the segmentation
head transplant instead of classifier rows:

```yaml
model:
  name: smp
  arch: unet
  encoder_name: resnet18
  checkpoint_path: runs/seg_v1/model.pt

tasks:
  mask:
    kind: segmentation
    target: mask_path
    head: native
    classes: {0: background, 1: defect, 2: edge, 3: scratch}   # scratch is novel
```

Refused loudly: a checkpoint whose feature space disagrees with the head's
input, more carried classes than the task declares (narrowing needs a mapping
this transplant does not guess), and classifier shapes that are not one
weight/bias pair.

One honest caveat, out of scope here by design: data labelled only for the
novel class pushes the old classes toward zero through plain BCE — when
"how to merge the datasets" becomes the question, start there.

## LoRA adapters

```yaml
defaults:
  - adapters: lora        # or: none

adapters:
  name: lora
  target_modules: [qkv, proj, fc1, fc2]
  rank: 8
  alpha: 16
```

Instead of training a large backbone's weights, train a small low-rank delta
beside each named projection and hold the rest still — measured on a timm
ViT-tiny, 118K trainable parameters out of 5.6M. Every remaining knob of peft's
`LoraConfig` (`use_dora`, `use_rslora`, `bias`, `exclude_modules`) forwards
verbatim.

`target_modules` has no default, because no architecture implies one. The names
are module-name suffixes or regexes — check them against the backbone's own
`named_modules()`. A value matching nothing is refused while the experiment is
built, since an unmatched target would silently train every weight at full
cost.

**The adapters own the backbone's freezing.** A `freeze` callback aimed at
`backbone` is refused: it would hold the adapters still too, and training
would run with nothing to learn.

**The deltas fold back before anything reads the weights.** After training — and
after the best checkpoint is restored — every adapter is merged into the layer it
stood in for. The fold is exact, so `test` evaluates what the artifact carries, and
the exported graph has no adapter overhead and no `lora_` names in it. A LoRA
run's output is indistinguishable from a plain run's.

**Warm starts.** `model.checkpoint_path` (a timm or smp file) loads while the
backbone is built, before the adapters go on, so starting from pretrained weights
needs nothing special. What does not work is `run.checkpoint_path` pointing at a
checkpoint from a run *without* adapters: injection renames every targeted
layer's weights, so the two are not interchangeable, and the mismatch is refused
by name rather than patched over.

## Distillation

```yaml
defaults:
  - distillation: kl      # or: none

distillation:
  teachers:
    - backbone: {name: timm, model_name: resnet50, pretrained: false}
      checkpoint_path: runs/teacher/best.ckpt
  loss: {name: kl_divergence, temperature: 2.0, weight: 0.7}
```

Each task's **training** loss gains a soft term comparing the student's logits
with the teachers' averaged ones. Validation and test are untouched — the
teachers do not even run there — so a distilled run's numbers stay comparable
with an undistilled one's, and a checkpoint monitor watching `val/loss` still
means what it meant.

**The soft term is an ordinary loss term.** `loss:` is the same declaration a
task's own `loss:` takes, so the temperature and the weight sit on it, the term
is logged under its own name beside the hard one (`train/label/kl` next to
`train/label/ce`), and a list declares several comparisons added with their
weights. There is no second place to weight distillation, and `weight: 0` is
refused rather than quietly training a teacher nobody listens to — drop the
section instead.

The teachers are built from **this run's tasks**, so their heads match the
student's output shapes by construction. `checkpoint_path` is optional: a
backbone declared `pretrained: true` is already a teacher. Several teachers are
a longer list; their raw logits are averaged, and one teacher is simply a list
of one.

**The artifact is the size of its student.** The teachers are held off the module
tree — registered, they would be written into every checkpoint and into the EMA
callback's copy of the module, and a traced graph carries whatever is registered.
Nothing about a deployment says a teacher was ever there.

**A checkpoint carries the weights of the model that ships.** A distilled run
writes its student under `model.student.…`, but `run.checkpoint_path` reads past
the scaffolding, so one file works either way: warm-start distillation from a
plain run's checkpoint, or point a later `train: false` run at a distilled one
without re-declaring teachers it will not use.

**Dot-paths do not move.** A `freeze` callback names modules relative to the model
that ships, so in a distilled run the backbone is still `backbone` — the student's — and
the guard that rejects `adapters` plus a `freeze` on the same backbone reads the same
word. A path that misses is refused with the children that were actually found.

**Distillation is not a callback, and cannot be.** No Lightning hook's return
value reaches the loss, and a callback adding its own backward pass would
contribute unscaled gradients against AMP's scaled ones — silently turning
`weight` into `weight / scale`. It changes what the model *computes*, which is
what earns a section of its own; a technique that only changes what the model
*holds* — which weights move, what is saved, what is averaged — is a callback.
